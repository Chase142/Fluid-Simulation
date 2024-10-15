import numpy as np
from scipy.spatial import cKDTree

def initialize(height, width, num_particles, radius, Rho, scale):
    meter_per_pixel =  scale / width

    height_m = height * meter_per_pixel
    width_m = width * meter_per_pixel
    radius_m = radius * meter_per_pixel

    positions = np.random.rand(num_particles, 2) * np.array([width-radius, height-radius])
    positions = positions * meter_per_pixel


    velocities = np.zeros((num_particles, 2))
    accelerations = np.zeros((num_particles, 2))


    r = radius * meter_per_pixel
    area = np.pi * (r**2)
    mass = Rho * area
    masses = np.ones(num_particles) * mass

    forces = np.zeros_like(positions)

    return (positions, velocities, accelerations, forces, masses, radius, height_m, width_m, radius_m, meter_per_pixel)

def normal_force(forces, positions, velocities, radius, height, width, k, c):
    x = positions[:, 0]
    y = positions[:, 1]

    touching_left = np.where(x - radius <= 0)[0]
    if touching_left.size > 0:
        n = np.array([1.0, 0.0])
        v_norm = np.dot(velocities[touching_left], n)
        delta = (x[touching_left] - radius) * -1

        delta = delta[:, np.newaxis]
        v_norm = v_norm[:, np.newaxis]

        forces[touching_left] += (k * delta * n) + (-c * v_norm * n)
    
    touching_right = np.where(x + radius >= width)[0]
    if touching_right.size > 0:
        n = np.array([-1.0, 0.0])
        v_norm = np.dot(velocities[touching_right], n)
        delta = x[touching_right] + radius - width

        delta = delta[:, np.newaxis]
        v_norm = v_norm[:, np.newaxis]

        forces[touching_right] += (k * delta * n) + (-c * v_norm * n)

    touching_top = np.where(y - radius <= 0)[0]
    if touching_top.size > 0:
        n = np.array([0.0, 1.0])
        v_norm = np.dot(velocities[touching_top], n)
        delta = (y[touching_top] - radius) * -1

        delta = delta[:, np.newaxis]
        v_norm = v_norm[:, np.newaxis]

        forces[touching_top] += (k * delta * n) + (-c * v_norm * n)

    touching_bottom = np.where(y + radius >= height)[0]
    if touching_bottom.size > 0:
        n = np.array([0.0, -1.0])
        v_norm = np.dot(velocities[touching_bottom], n) 
        delta = y[touching_bottom] + radius - height

        delta = delta[:, np.newaxis]
        v_norm = v_norm[:, np.newaxis]

        forces[touching_bottom] += (k * delta * n) + (-c * v_norm * n)

def gravity(forces, masses, g):
    forces[:,1] += (g*masses)


def global_velocity_damping(velocities, damping_factor=0.99):
    velocities *= damping_factor
    return velocities

def collisions(positions, velocities, forces, masses, radius, restitution=0.5, k=10000, epsilon=1e-6):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=2*radius)

    for i, j in pairs:
        delta_pos = positions[j] - positions[i]
        distance = np.linalg.norm(delta_pos)
        repulsive_force(i, j, forces, positions, radius, k, epsilon)
        if distance == 0:
            continue
        delta = (2 * radius) - distance
        if delta > 0:
            n_ij = delta_pos / (distance + epsilon)

            delta_vel = velocities[j] - velocities[i]
            v_rel = np.dot(delta_vel, n_ij)

            if v_rel < 0:
                impulse = (1 + restitution) * v_rel / (1 / masses[i] + 1 / masses[j])

                impulse_vector = impulse * n_ij

                velocities[i] += impulse_vector / masses[i]
                velocities[j] -= impulse_vector / masses[j]

    return velocities

def repulsive_force(i, j, forces, positions, radius, k_repulsion=10000, epsilon=1e-6):
    delta_pos = positions[j] - positions[i]
    distance = np.linalg.norm(delta_pos)

    if distance < 2 * radius:
        n_ij = delta_pos / (distance + epsilon)
        overlap = 2 * radius - distance

        force_magnitude = k_repulsion * overlap
        force_vector = force_magnitude * n_ij

        forces[i] -= force_vector
        forces[j] += force_vector

    return forces

def user_force(forces, window_position, dt, window_mass, epsilon=1-6):
    window_acceleration = (window_position[2] - 2 * window_position[1] + window_position[0]) / dt**2

    delta_pos = window_position[2] - window_position[0]
    distance = np.linalg.norm(delta_pos)
    direction = delta_pos / (distance + epsilon)

    F = -1 * (window_acceleration / window_mass) * direction
    forces += F
    return forces


def sum_forces(forces, positions, velocities, masses, radius, height, width, g, k, c):

    gravity(forces, masses, g)

    normal_force(forces, positions, velocities, radius, height, width, k, c)

    return forces

def update(positions, velocities, masses, forces, dt):
    accelerations = forces / masses[:, np.newaxis]
    positions += velocities * dt + 0.5 * accelerations * dt**2

    velocities += accelerations * dt

    forces = np.zeros_like(positions)

    return positions, velocities, forces


def normal_force_obj(force, position, velocity, size, height, width, k, c):
    x = position[0]
    y = position[1]

    h_x = size[0]
    h_y = size[1]

    if x <= 0:
        n = np.array([1.0, 0.0])
        v_norm = np.dot(velocity, n)
        delta = -x  
        force += (k * delta * n) - (c * v_norm * n)

    if x + h_x >= width:
        n = np.array([-1.0, 0.0])
        v_norm = np.dot(velocity, n)
        delta = x + h_x - width
        force += (k * delta * n) - (c * v_norm * n)

    if y <= 0:
        n = np.array([0.0, 1.0])
        v_norm = np.dot(velocity, n)
        delta = -y
        force += (k * delta * n) - (c * v_norm * n)

    if y + h_y >= height:
        n = np.array([0.0, -1.0])
        v_norm = np.dot(velocity, n)
        delta = y + h_y - height
        force += (k * delta * n) - (c * v_norm * n)

def particle_box_interactions(positions, velocities, masses, obj_positions, obj_velocities, obj_masses, obj_sizes, radius_m, restitution=0.5, epsilon=1e-6):
    for box_idx in range(len(obj_positions)):
        box_pos = obj_positions[box_idx]
        box_vel = obj_velocities[box_idx]
        box_mass = obj_masses[box_idx]
        box_size = obj_sizes[box_idx]
        
        box_x_min = box_pos[0]
        box_x_max = box_pos[0] + box_size[0]
        box_y_min = box_pos[1]
        box_y_max = box_pos[1] + box_size[1]
        
        for i in range(len(positions)):
            particle_pos = positions[i]
            particle_vel = velocities[i]
            particle_mass = masses[i]
            
            if (box_x_min - radius_m <= particle_pos[0] <= box_x_max + radius_m and
                box_y_min - radius_m <= particle_pos[1] <= box_y_max + radius_m):
                
                closest_x = np.clip(particle_pos[0], box_x_min, box_x_max)
                closest_y = np.clip(particle_pos[1], box_y_min, box_y_max)
                closest_point = np.array([closest_x, closest_y])
                
                delta_pos = particle_pos - closest_point
                distance = np.linalg.norm(delta_pos)
                penetration_depth = radius_m - distance
                
                if penetration_depth > 0:
                    n = delta_pos / (distance + epsilon)
                    
                    relative_velocity = particle_vel - box_vel
                    v_rel = np.dot(relative_velocity, n)
                    
                    total_mass = particle_mass + box_mass
                    impulse_magnitude = -(1 + restitution) * v_rel / (1 / particle_mass + 1 / box_mass)
                    
                    impulse = impulse_magnitude * n
                    velocities[i] += impulse / particle_mass
                    obj_velocities[box_idx] -= impulse / box_mass
                    
                    correction_factor = 0.5  
                    correction = correction_factor * penetration_depth * n
                    positions[i] += correction
                    obj_positions[box_idx] -= correction
                    
    return velocities, obj_velocities


def sum_forces_obj(obj_forces, obj_positions, obj_velocities, obj_masses, obj_sizes, height, width, k, c, g):
    for i in range(len(obj_positions)):
        obj_forces[i] = np.zeros(2)
        obj_forces[i][1] += g * obj_masses[i]
        normal_force_obj(obj_forces[i], obj_positions[i], obj_velocities[i], obj_sizes[i], height, width, k, c)
    return obj_forces, obj_velocities

def update_objs(obj_positions, obj_velocities, obj_forces, obj_masses, dt):
    obj_positions = np.array(obj_positions)
    obj_velocities = np.array(obj_velocities)
    obj_forces = np.array(obj_forces)
    obj_masses = np.array(obj_masses)
    
    for i in range(len(obj_positions)):
        force = obj_forces[i]
        mass = obj_masses[i]
        
        acceleration = force / mass  
        acceleration = acceleration.reshape(-1) 

        obj_positions[i] += obj_velocities[i] * dt + 0.5 * acceleration * dt**2
        obj_velocities[i] += acceleration * dt

        obj_forces[i] = np.zeros(2)
    
    return obj_positions.tolist(), obj_velocities.tolist(), obj_forces.tolist()