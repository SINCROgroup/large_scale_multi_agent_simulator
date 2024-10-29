# Test commit


class Simulator():

    def __init__(self,agents,env,controller,integrator,logger,render,conf_file) -> None:

        #get parameters from initialization 
        self.agents = agents
        self.env = env
        self.controller = controller
        self.logger = logger
        self.render = render
        self.integrator = integrator





    def simulate(self):


        for t in range(0, self.t_f, self.dt):

            u = self.controller.get_action(self.agents.x,self.env.x)
            f = self.env.get_forces(self.agents.x)
            
            self.agents.x = self.integrator.step(self.agents.x, u, f)
            # Update the environment

            #Execute every N steps
            self.logger.log(self.agents.x,u,f,self.env)
            self.render.render(self.agents.x,u,f,self.env)



            






