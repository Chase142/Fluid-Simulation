import pygame
import time
from util import *
from AppKit import NSApplication, NSApp, NSWindow

FPS = 200

# scale 1m by 1m
scale = 1

N = 500
WIDTH, HEIGHT = 800, 800
RADIUS = 5

obj_mass = 10  #kg
obj_size = np.array([50.0, 30.0])  

window_mass = 100000

Rho = 10000 # kg/m^3

k_wall = 50000  # N/m
c_wall = 100  # kg/s 

g = 9.8 #m/s^2

mu = 1 # velocity damping factor

k_repulsion = 50000 # N/m

restitution_particles = .8 # 0 for perfectly inelastic, 1 for elastic
restitution_box = 1

positions, velocities, accelerations, forces, masses, radius, height_m, width_m, radius_m, meter_per_pixel = initialize(HEIGHT, WIDTH, N, RADIUS, Rho, scale)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Fluid Simulation")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 36)

app = NSApplication.sharedApplication()

def get_window_position():
    window = NSApp().keyWindow()
    if window is not None:
        frame = window.frame()
        x, y = frame.origin.x, frame.origin.y
        return x, y
    return None, None

click_start_time = None
click_duration = 0

obj_sizes = []
obj_positions = []  
obj_velocities = []  
obj_forces = [] 
obj_masses = [] 
objs = 0

last_window_pos = np.array([0.0,0.0])
window_position = np.zeros((3, 2))
i = 0
running = True
while running:
    clock.tick(FPS)
    dt = 0.001 # seconds
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click_start_time = time.time()
        if event.type == pygame.MOUSEBUTTONUP:
            if click_start_time is not None:
                click_duration = time.time() - click_start_time  
                objs += 1
                mouse_x, mouse_y = pygame.mouse.get_pos()
                rect_position = np.array([mouse_x*meter_per_pixel, mouse_y*meter_per_pixel])
                obj_positions.append(rect_position)
                obj_masses.append(obj_mass*click_duration*2)
                obj_sizes.append(obj_size*meter_per_pixel*click_duration*2)
                obj_velocities.append(np.zeros(2))
                obj_forces.append(np.zeros(2))
                click_start_time = None  

    temp = get_window_position()
    if temp[0] is not None:
        window_position[i] = temp
        last_window_pos = temp
        i += 1
    else:
        window_position[i] = last_window_pos
    if i == 3:
        forces = user_force(forces, window_position, dt, window_mass)
        window_position.fill(0)
        i = 0


    velocities = collisions(positions, velocities, forces, masses, radius_m, restitution_particles, k_repulsion)

    if objs > 0:
        velocities, obj_velocities = particle_box_interactions(positions, velocities, masses, obj_positions, obj_velocities, obj_masses, obj_sizes, radius_m, restitution_box)
    
    velocities = global_velocity_damping(velocities, mu)

    forces = sum_forces(forces, positions, velocities, masses, radius_m, height_m, width_m, g, k_wall, c_wall)

    positions, velocities, forces = update(positions, velocities, masses, forces, dt)
    
    if objs > 0:
        obj_forces, obj_velocities = sum_forces_obj(obj_forces, obj_positions, obj_velocities, obj_masses, obj_sizes, height_m, width_m, k_wall, c_wall, g)
        obj_positions, obj_velocities, obj_forces = update_objs(obj_positions, obj_velocities, obj_forces, obj_masses, dt)
    
    # Clear screen
    screen.fill((0, 0, 0))

    # Background
    background_color = (0, 0, 0)
    screen.fill(background_color)
    
    # Draw particles
    for pos in positions:
        pos_pixels = (pos / meter_per_pixel).astype(int)
        pygame.draw.circle(screen, (0, 255, 255), pos_pixels, radius)
    for j in range(len(obj_positions)):
        pos_pixels = (np.array(obj_positions[j]) / meter_per_pixel).astype(int)
        size_pixels = (np.array(obj_sizes[j]) / meter_per_pixel).astype(int)
        pygame.draw.rect(screen, (238,210,10), np.append(pos_pixels, size_pixels))

    fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))
    
    # Update display
    pygame.display.flip()
pygame.quit()