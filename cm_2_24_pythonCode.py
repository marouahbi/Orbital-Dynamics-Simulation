import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67e-11  # The gravitational constant in N m^2/kg^2
M_SUN = 1.989e30  # The mass of the Sun in kg

""" The planetary data is stored in a dictionary to easily access the relevant properties.
    This way, when a planet's name is entered, we can retrieve its mass, radius in AU, and eccentricity. """
planet_data = {
    "Mercury": {"mass": 2.4e23, "radius_au": 0.39, "eccentricity": 0.206},
    "Venus": {"mass": 4.9e24, "radius_au": 0.72, "eccentricity": 0.007},
    "Earth": {"mass": 6.0e24, "radius_au": 1.0, "eccentricity": 0.017},
    "Mars": {"mass": 6.6e23, "radius_au": 1.52, "eccentricity": 0.093},
    "Jupiter": {"mass": 1.9e27, "radius_au": 5.2, "eccentricity": 0.048},
    "Saturn": {"mass": 5.7e26, "radius_au": 9.54, "eccentricity": 0.056},
    "Uranus": {"mass": 8.8e25, "radius_au": 19.19, "eccentricity": 0.046},
    "Neptune": {"mass": 1.0e26, "radius_au": 30.06, "eccentricity": 0.010},
    "Pluto": {"mass": 1.3e22, "radius_au": 39.26, "eccentricity": 0.248},
}


def get_planet_data(planet_name):
    """Retrieve the mass, radius, and eccentricity of a given planet.
    The function then takes the planet's name as input and returns its properties from the dictionary.
    """
    return planet_data.get(planet_name)


def initialize_conditions(m_planet, r_au, eccentricity, dt, t_max):
    """Initialize the simulation conditions for the planet's motion.

    The initial position is set based on the planet's radius in AU then it is converted to meters.
    The velocity is computed for circular motion, adjusted by the planet's eccentricity.
    We calculate how many time steps are needed based on the total time and time step size.
    """
    r_planet = r_au * 1.496e11  # Convert AU to meters.

    # Calculate the circular orbital velocity using the gravitational force formula.
    v_circular = np.sqrt(G * M_SUN / r_planet)

    # Modifying initial velocity based on eccentricity of the planet
    v_planet = v_circular * (1 + eccentricity)

    # Number of steps of the whole simulation
    num_steps = int(t_max / dt)

    # Initialize position and velocity arrays for each time step.
    positions = np.zeros((num_steps, 2))
    velocities = np.zeros((num_steps, 2))

    # Simulation starts off with the initial velocities and positions
    positions[0] = [r_planet, 0]  # Initial position in x-direction, with y as zero.
    velocities[0] = [0, v_planet]  # Initial velocity is in the y-direction.

    return positions, velocities, num_steps


def compute_gravitational_force(planet_pos):

    """Calculation of the gravitational force acting on the planet by the Sun.
    The force is determined using Newton's law of gravitation, with the force vector points towards the Sun.
    """

    r = np.linalg.norm(planet_pos)  # Distance from the sun.
    return -G * M_SUN * planet_pos / r ** 3  # Gravitational force vector.


def energy_calc(positions, velocities, dt, num_steps, m_planet):

    """Calculate kinetic, potential, and total energy over time.
    The function loops through each time step, updating position and velocity,
    while calculating the energies based on the current state of the planet.
    """

    kinetic_energy = np.zeros(num_steps)
    potential_energy = np.zeros(num_steps)
    total_energy = np.zeros(num_steps)

    for step in range(num_steps - 1):
        # Compute the gravitational force at the current position of the planet.
        F_gravity = compute_gravitational_force(positions[step])

        # Update the planet's velocity based on the gravitational force.
        velocities[step + 1] = velocities[step] + F_gravity / m_planet * dt

        # Update the planet's position based on its new velocity.
        positions[step + 1] = positions[step] + velocities[step] * dt

        # Calculate kinetic energy using the formula KE = 0.5 * m * v^2.
        kinetic_energy[step] = 0.5 * m_planet * np.linalg.norm(velocities[step]) ** 2

        # Compute the distance for potential energy calculations.
        r = np.linalg.norm(positions[step])

        # Calculate potential energy using PE = -G * m1 * m2 / r.
        potential_energy[step] = -G * m_planet * M_SUN / r

        # Total energy is the sum of kinetic and potential energy.
        total_energy[step] = kinetic_energy[step] + potential_energy[step]

    # Explicitly calculate the energies for the last time step.
    kinetic_energy[-1] = 0.5 * m_planet * np.linalg.norm(velocities[-1]) ** 2
    r = np.linalg.norm(positions[-1])
    potential_energy[-1] = -G * m_planet * M_SUN / r
    total_energy[-1] = kinetic_energy[-1] + potential_energy[-1]

    return kinetic_energy, potential_energy, total_energy


def energy_change_calc(total_energy):

    """Calculation of the relative change in energy over time to evaluate conservation.
    This function computes how much the total energy deviates from its initial value,
    which helps in analyzing the accuracy of the simulation.
    """
    initial_energy = total_energy[0]  # Initial energy.
    energy_change = (total_energy - initial_energy) / np.abs(initial_energy)  # Relative energy change.
    return energy_change


# User inputs
planet_name = input("Enter the name of the planet (e.g., Earth, Uranus): ")
planet_info = get_planet_data(planet_name)

if planet_info:
    # Extraction of mass, radius, and eccentricity of the specified planet.
    m_planet = planet_info["mass"]  # Mass in kg.
    r_au = planet_info["radius_au"]  # Radius in AU.
    eccentricity = planet_info["eccentricity"]  # Eccentricity of the orbit.

    # Total simulation time in seconds (one year).
    t_max = 365 * 24 * 3600

    # Define the time steps.
    dt_values = [60 * 60 * 24 * i for i in range(1, 31)]  # Time steps from 1 day to 30 days.

    energy_changes = []

    # Loop over the different time step values to analyze energy conservation.
    for dt in dt_values:
        positions, velocities, num_steps = initialize_conditions(m_planet, r_au, eccentricity, dt, t_max)

        _, _, total_energy = energy_calc(positions, velocities, dt, num_steps, m_planet)

        energy_change = energy_change_calc(total_energy)
        energy_changes.append(energy_change[-1])  # Final energy change.

    # Plotting Results
    plt.figure(figsize=(12, 6))
    plt.plot(dt_values, energy_changes, marker='o', label=f'Energy Change for {planet_name}')

    plt.xscale('log')
    plt.title("Energy Change Over One Orbit vs Time Step (Δt)")
    plt.xlabel("Time Step (s)")
    plt.ylabel("Relative Energy Change")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Planet not found. Please check the name and try again.")
