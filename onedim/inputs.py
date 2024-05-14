import configparser
import numpy as np
import os


class CaseInsensitiveConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optionxform = (
            str.lower
        )  # Transform options to lowercase for case insensitivity


class Inputs:
    def __init__(self, fname):
        config = CaseInsensitiveConfigParser()
        config.read(fname)

        if not os.path.exists(fname):
            raise RuntimeError("Inputs File not found.")

        # Mesh
        self.nx = self.get_config_value(config, "Mesh", "nx", type_func=int)
        self.numghosts = self.get_config_value(
            config, "Mesh", "numghosts", type_func=int
        )
        self.x_lo = self.get_config_value(config, "Mesh", "x_lo", type_func=float)
        self.x_hi = self.get_config_value(config, "Mesh", "x_hi", type_func=float)

        self.xlim = (self.x_lo, self.x_hi)

        # Time
        self.time_steps = self.get_config_value(
            config, "Time", "time_steps", type_func=int, mandatory=False, default=np.inf
        )
        self.method = self.get_config_value(config, "Time", "method")
        self.t0 = self.get_config_value(
            config, "Time", "t0", mandatory=False, default=0.0, type_func=float
        )
        self.t_finish = self.get_config_value(
            config, "Time", "t_finish", mandatory=False, default=np.inf, type_func=float
        )

        if (self.t_finish == np.inf) and (self.time_steps == np.inf):
            raise RuntimeError(
                "Must supply either t_finish or time_steps in [Time] section of inputs file."
            )

        if self.t0 >= self.t_finish:
            raise RuntimeError("Initial time is >= to the final time.")

        # Fluid
        self.ics = self.get_config_value(config, "Fluid", "ics")

        # Method
        self.flux = self.get_config_value(config, "Method", "flux")
        self.bc_lo = self.get_config_value(config, "Method", "bcs_lo")
        self.bc_hi = self.get_config_value(config, "Method", "bcs_hi")

        # Output
        self.output_freq = self.get_config_value(
            config, "Output", "output_freq", type_func=int, mandatory=False, default=1
        )
        self.output_dir = self.get_config_value(
            config, "Output", "output_dir", mandatory=False, default="output/"
        )
        self.make_movie = self.get_config_value(
            config,
            "Output",
            "make_movie",
            type_func=bool,
            mandatory=False,
            default=True,
        )

        # nodes_z = get_config_value(config, 'Mesh', 'nodes_z', mandatory=False, default=50)

        # # Accessing the data from the config
        # time_steps = config.getint('Simulation', 'time_steps')
        # delta_t = config.getfloat('Simulation', 'delta_t')
        # method = config.get('Simulation', 'method')

        # density = config.getfloat('Fluid', 'density')
        # viscosity = config.getfloat('Fluid', 'viscosity')

        # mesh_type = config.get('Mesh', 'type')
        # nodes_x = config.getint('Mesh', 'nodes_x')
        # nodes_y = config.getint('Mesh', 'nodes_y')

        # # Printing the values
        # print("Simulation Parameters:")
        # print("Time Steps:", time_steps)
        # print("Delta Time:", delta_t)
        # print("Method:", method)

        # print("\nFluid Properties:")
        # print("Density:", density)
        # print("Viscosity:", viscosity)

        # print("\nMesh Configuration:")
        # print("Type:", mesh_type)
        # print("Nodes in X:", nodes_x)
        # print("Nodes in Y:", nodes_y)

    def get_config_value(
        self, config, section, option, type_func=str, mandatory=True, default=None
    ):
        try:
            # Attempt to get the value with type conversion; if not found or conversion fails, it will raise an error
            value = config.get(
                section, option, fallback=default if not mandatory else None
            )
            if value is None and mandatory:
                raise ValueError(
                    f"Missing mandatory argument: '{option}' in section '{section}'"
                )
            return type_func(value)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            if mandatory:
                raise ValueError(
                    f"Missing mandatory argument: '{option}' in section '{section}'"
                ) from e
            return default
        except ValueError as e:
            raise ValueError(
                f"Type conversion error for '{option}' in section '{section}': {e}"
            ) from e
