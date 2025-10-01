import pathlib
from datetime import datetime
from swarmsim.Loggers import Logger
from swarmsim.Utils.sim_utils import load_config
import numpy as np
import scipy.io as sio
import time
import csv


class BaseLogger(Logger):
    def __init__(self, populations: list, environment: object, config_path: str, buffer_size: int = 10) -> None:
        super().__init__()

        # Load configuration file
        config: dict = load_config(config_path)
        class_name = type(self).__name__
        self.config = config
        self.logger_config = config.get(class_name, {})

        # Logger parameters
        now = datetime.now()
        self.timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
        self.log_id = now.strftime('%Y%m%d_%H%M%S') + self.logger_config.get('log_name', '')
        self.enabled = self.logger_config.get('activate', True)
        self.print_freq_step = self.logger_config.get('print_freq_step', 0)
        self.save_freq_step = self.logger_config.get('save_freq_step', 0)
        self.print_freq_exp = self.logger_config.get('print_freq_experiment', 0)
        self.save_freq_exp = self.logger_config.get('save_freq_experiment', 0)
        self.comment_enabled = self.logger_config.get('comment_enable', False)
        self.buffer_size = buffer_size

        # Setup folders and file names
        log_dir = pathlib.Path(self.logger_config.get('log_path', './logs')) / self.log_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self.base_path = log_dir / self.log_id

        # Experiment state
        self.populations = populations
        self.environment = environment
        self.start_time = None
        self.end_time = None
        self.step_idx = None
        self.exp_idx = -1
        self.finished = False
        self.step_data = {}
        self.exp_data = {}

        # Buffers
        self.step_buffer = []  # rows for current experiment
        self.exp_buffer = []   # rows for all experiments

        if self.enabled:
            comment = input('Comment: ') if self.comment_enabled else ''
            with open(self.base_path.with_suffix('.txt'), 'w') as f:
                f.write(f"Date: {self.timestamp_str}\n")
                f.write("Configuration settings:\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"\nInitial comment: {comment}")

    def reset(self) -> bool:
        """Start a new experiment run."""
        self.finished = False
        self.step_idx = 0
        self.exp_idx += 1
        self.exp_data = {}
        self.step_buffer = []  # reset step buffer for new experiment
        if self.enabled:
            self.start_time = time.time()
        return self.enabled

    def log(self, data: dict | None = None, end_experiment: bool = False):
        """Log one step (and optionally the experiment summary)."""
        self.step_data = {}
        if self.enabled:
            self.log_step_data()
            self._log_external(data, step=True)
            if self.save_freq_step > 0 and self.step_idx % self.save_freq_step == 0:
                self._maybe_buffer(step=True)
            if self.print_freq_step > 0 and self.step_idx % self.print_freq_step == 0:
                self.print_log(step=True)

            if end_experiment:
                self.exp_data = {}
                self.log_experiment_data()
                self._log_external(data, step=False)
                if self.save_freq_exp > 0 and self.exp_idx % self.save_freq_exp == 0:
                    self._maybe_buffer(step=False)
                self._flush_buffers()  # flush everything when experiment ends
                if self.print_freq_exp > 0 and self.exp_idx % self.print_freq_exp == 0:
                    self.print_log(step=False)

        self.step_idx += 1
        return self.finished

    def close(self, data: dict | None = None) -> bool:
        """Close the logger, logging the final step and experiment data."""
        print('Simulation completed. Saving...')
        if self.enabled:
            comment = input('\nComment: ') if self.comment_enabled else ''
            with open(self.base_path.with_suffix('.txt'), 'a') as f:
                elapsed = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
                f.write(f"\nElapsed time [s]: {elapsed}\nComments: {comment}\n")
            self.end_time = time.time()
            self.finished = self.log(data=data, end_experiment=True)
            self._flush_buffers(force=True)
            self._export_npz_mat()
        return self.enabled

    def log_step_data(self):
        """
        Hook for subclasses to add step-level data.
        Example:
            self._add_entry(('csv', 'print'), step=True, reward=current_reward)
        """
        self._add_entry(('csv', 'print'), step=True, step_count=self.step_idx)

    def log_experiment_data(self):
        """
        Hook for subclasses to add experiment-level data.
        Example:
            self._add_entry(('csv', 'txt'), step=False, avg_reward=np.mean(rewards))
        """
        self._add_entry(('csv', 'print', 'mat', 'npz'), step=False, done=self.finished)  # Add done flag

    # ---------------- INTERNAL HELPERS ---------------- #

    def _log_external(self, data: dict | None, step: bool, save_modes=('csv', 'npz', 'mat')):
        if not data:
            return
        for key, value in data.items():
            self._add_entry(save_modes, step=step, **{key: value})

    def _maybe_buffer(self, step: bool):
        """Add entry to buffer, and flush if buffer is full."""
        if step:
            row = {}
            for k, v in self.step_data.items():
                if 'csv' in v['save_modes']:
                    val = v['value']
                    if isinstance(val, np.ndarray):
                        flat = val.ravel().astype(float)
                        row[k] = ",".join(str(x) for x in flat)  # ✅ no brackets
                    else:
                        row[k] = str(val)
            if row:
                self.step_buffer.append(row)
            if len(self.step_buffer) >= self.buffer_size:
                self._flush_step_buffer()
        else:
            row = {}
            for k, v in self.exp_data.items():
                if 'csv' in v['save_modes']:
                    val = v['value']
                    if isinstance(val, np.ndarray):
                        flat = val.ravel().astype(float)
                        row[k] = ",".join(str(x) for x in flat)
                    else:
                        row[k] = str(val)
            if row:
                self.exp_buffer.append(row)

    def _flush_step_buffer(self):
        """Flush step buffer to CSV for current experiment."""
        if not self.step_buffer:
            return

        csv_path = self.base_path.with_name(f"{self.base_path.name}_exp_{self.exp_idx}.csv")
        write_header = not csv_path.exists()

        def serialize_value(v):
            """Convert scalars/arrays/bools/strings to clean CSV string."""
            if isinstance(v, (np.ndarray, list, tuple)):
                return ",".join(str(serialize_value(x)) for x in v)
            elif isinstance(v, (float, np.floating, int, np.integer)):
                return str(v)
            elif isinstance(v, (bool, np.bool_)):
                return "true" if v else "false"
            elif isinstance(v, str):
                return v  # no quotes
            else:
                return str(v)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.step_buffer[0].keys())
            if write_header:
                writer.writeheader()

            for row in self.step_buffer:
                clean_row = {k: serialize_value(v) for k, v in row.items()}
                writer.writerow(clean_row)

        self.step_buffer.clear()

    def _flush_exp_buffer(self):
        """Flush experiment buffer to global CSV."""
        if not self.exp_buffer:
            return

        csv_path = self.base_path.with_name(f"{self.base_path.name}_experiments.csv")
        write_header = not csv_path.exists()

        def serialize_value(v):
            """Convert scalars/arrays/bools/strings to clean CSV string."""
            if isinstance(v, (np.ndarray, list, tuple)):
                return ",".join(str(serialize_value(x)) for x in v)
            elif isinstance(v, (float, np.floating, int, np.integer)):
                return str(v)
            elif isinstance(v, (bool, np.bool_)):
                return "true" if v else "false"
            elif isinstance(v, str):
                return v  # no quotes
            else:
                return str(v)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.exp_buffer[0].keys())
            if write_header:
                writer.writeheader()

            for row in self.exp_buffer:
                clean_row = {k: serialize_value(v) for k, v in row.items()}
                writer.writerow(clean_row)

        self.exp_buffer.clear()

    def _flush_buffers(self, force: bool = False):
        """Flush both step and experiment buffers."""
        self._flush_step_buffer()
        self._flush_exp_buffer()
        if force:
            self.step_buffer.clear()
            self.exp_buffer.clear()

    def _add_entry(self, save_modes=(), step: bool = True, **kwargs):
        target = self.exp_data if not step else self.step_data
        for key, value in kwargs.items():
            target[key] = {'value': value, 'save_modes': tuple(save_modes)}

    def _export_npz_mat(self):
        """Export CSV data into .npz and .mat formats as dicts of arrays.

        Only exports variables whose `save_modes` include 'npz' or 'mat'.
        """

        def try_cast(val: str):
            v = val.strip().strip('"').strip("'")
            if v.lower() in ("true", "false"):
                return v.lower() == "true"
            try:
                return float(v)
            except ValueError:
                return v

        def parse_entry(entry: str):
            entry = entry.strip().strip('"').strip("'")
            if not entry:
                return np.array([])
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            if len(parts) == 1:
                return try_cast(parts[0])
            else:
                return np.array([try_cast(x) for x in parts], dtype=object)

        # Determine which columns should be exported
        def get_allowed_fields():
            allowed_npz, allowed_mat = set(), set()
            for data_dict in (self.step_data, self.exp_data):
                for k, v in data_dict.items():
                    modes = v.get("save_modes", ())
                    if "npz" in modes:
                        allowed_npz.add(k)
                    if "mat" in modes:
                        allowed_mat.add(k)
            return allowed_npz, allowed_mat

        allowed_npz, allowed_mat = get_allowed_fields()

        experiments_csv = self.base_path.with_name(f"{self.base_path.name}_experiments.csv")
        if experiments_csv.exists():
            with open(experiments_csv, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)  # column names
                rows = list(reader)

            T = len(rows)
            raw_data = {name: [] for name in header}

            for row in rows:
                for col_name, cell in zip(header, row):
                    raw_data[col_name].append(parse_entry(cell))

            data_dict = {}
            for name, values in raw_data.items():
                first = values[0]

                if isinstance(first, (float, np.floating)):
                    arr = np.array(values, dtype=float)
                elif isinstance(first, (bool, np.bool_)):
                    arr = np.array(values, dtype=bool)
                elif isinstance(first, str):
                    arr = np.array(values, dtype=object)
                elif isinstance(first, np.ndarray):
                    N = len(first)
                    if all(isinstance(v[i], (float, np.floating)) for v in values for i in range(len(v))):
                        arr = np.zeros((T, N), dtype=float)
                    elif all(isinstance(v[i], (bool, np.bool_)) for v in values for i in range(len(v))):
                        arr = np.zeros((T, N), dtype=bool)
                    else:
                        arr = np.empty((T, N), dtype=object)
                    for t in range(T):
                        if len(values[t]) != N:
                            raise ValueError(
                                f"Inconsistent length in column {name} at timestep {t}: "
                                f"expected {N}, got {len(values[t])}"
                            )
                        arr[t, :] = values[t]
                else:
                    arr = np.array(values, dtype=object)

                data_dict[name] = arr

            # --- Filter based on save_modes ---
            data_npz = {k: v for k, v in data_dict.items() if k in allowed_npz}
            data_mat = {k: v for k, v in data_dict.items() if k in allowed_mat}

            # Save NPZ
            if data_npz:
                np.savez(self.base_path.with_name(f"{self.base_path.name}_experiments.npz"), **data_npz)

            # Save MAT (convert object arrays to string arrays for MATLAB)
            if data_mat:
                mat_dict = {}
                for k, v in data_mat.items():
                    if v.dtype == object:
                        mat_dict[k] = np.array(v, dtype="U")
                    else:
                        mat_dict[k] = v
                sio.savemat(self.base_path.with_name(f"{self.base_path.name}_experiments.mat"), mat_dict)

        for exp in range(self.exp_idx + 1):
            step_csv = self.base_path.with_name(f"{self.base_path.name}_exp_{exp}.csv")
            if not step_csv.exists():
                continue

            with open(step_csv, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)   # column names
                rows = list(reader)

            T = len(rows)
            raw_data = {name: [] for name in header}

            for row in rows:
                for col_name, cell in zip(header, row):
                    raw_data[col_name].append(parse_entry(cell))

            data_dict = {}
            for name, values in raw_data.items():
                first = values[0]

                if isinstance(first, (float, np.floating)):
                    arr = np.array(values, dtype=float)
                elif isinstance(first, (bool, np.bool_)):
                    arr = np.array(values, dtype=bool)
                elif isinstance(first, str):
                    arr = np.array(values, dtype=object)
                elif isinstance(first, np.ndarray):
                    N = len(first)
                    if all(isinstance(v[i], (float, np.floating)) for v in values for i in range(len(v))):
                        arr = np.zeros((T, N), dtype=float)
                    elif all(isinstance(v[i], (bool, np.bool_)) for v in values for i in range(len(v))):
                        arr = np.zeros((T, N), dtype=bool)
                    else:
                        arr = np.empty((T, N), dtype=object)
                    for t in range(T):
                        if len(values[t]) != N:
                            raise ValueError(
                                f"Inconsistent length in column {name} at timestep {t}: "
                                f"expected {N}, got {len(values[t])}"
                            )
                        arr[t, :] = values[t]
                else:
                    arr = np.array(values, dtype=object)

                data_dict[name] = arr

            # --- Filter based on save_modes ---
            data_npz = {k: v for k, v in data_dict.items() if k in allowed_npz}
            data_mat = {k: v for k, v in data_dict.items() if k in allowed_mat}

            # Save NPZ
            if data_npz:
                np.savez(self.base_path.with_name(f"{self.base_path.name}_exp_{exp}.npz"), **data_npz)

            # Save MAT (convert object arrays to string arrays for MATLAB)
            if data_mat:
                mat_dict = {}
                for k, v in data_mat.items():
                    if v.dtype == object:
                        mat_dict[k] = np.array(v, dtype="U")
                    else:
                        mat_dict[k] = v
                sio.savemat(self.base_path.with_name(f"{self.base_path.name}_exp_{exp}.mat"), mat_dict)

    def print_log(self, step=True):
        if step:
            for key, value in self.step_data.items():
                if 'print' in value['save_modes']:
                    print(f"{key}: {value['value']}; ", end=" ")
            print('\n')
        else:
            for key, value in self.exp_data.items():
                if 'print' in value['save_modes']:
                    print(f"{key}: {value['value']}; ", end=" ")
            print('\n')
