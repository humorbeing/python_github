import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import random

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),

    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)

edges = (
    (0, 1),
    (0, 3),
    (0, 4),

    (2, 1),
    (2, 3),
    (2, 7),

    (6, 3),
    (6, 4),
    (6, 7),

    (5, 1),
    (5, 4),
    (5, 7),
)

surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),

    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),

    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)


def set_vertices(max_distance, min_distance=-20,
                 camera_x=0, camera_y=0):

    camera_x = -1*int(camera_x)
    camera_y = -1*int(camera_y)

    x_value_change = random.randrange(camera_x-10, camera_x+10)
    y_value_change = random.randrange(camera_y-10, camera_y+10)
    z_value_change = random.randrange(-1*max_distance, min_distance)

    new_vertices = []

    for vert in vertices:
        new_vert = []

        new_x = vert[0] + x_value_change
        new_y = vert[1] + y_value_change
        new_z = vert[2] + z_value_change

        new_vert.append(new_x)
        new_vert.append(new_y)
        new_vert.append(new_z)

        new_vertices.append(new_vert)

    return new_vertices


def Cube(vertices):
    glBegin(GL_QUADS)
    # glColor3fv((0, 1, 0))
    # x = 0  # set xxx
    for surface in surfaces:
        x = 0  # set yyy
        # x += 1  # set xxx
        # glColor3fv((0, 1, 0))
        for vertex in surface:
            x += 1  # set yyy
            glColor3fv(colors[x])

            glVertex3fv(vertices[vertex])
    # for edge in edges:
    #     for vertex in edge:
    #         glVertex3fv(vertices[vertex])
    glEnd()


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    max_distance = 100

    gluPerspective(45, (display[0]/display[1]), 0.1, max_distance)
    # gluPerspective(45, (display[0] / display[1]), 10, 20)

    glTranslatef(0, 0, -40)
    # glTranslatef(1, 1, -5)

    # glRotatef(0, 0, 0, 0)
    # glRotatef(25, 2, 0, 0)
    # object_passed = False

    x_move = 0
    y_move = 0

    cur_x = 0
    cur_y = 0

    cube_dict = {}

    for x in range(100):
        cube_dict[x] = set_vertices(max_distance)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_move = 0.2
                if event.key == pygame.K_RIGHT:
                    x_move = -0.2

                if event.key == pygame.K_UP:
                    y_move = -0.2
                if event.key == pygame.K_DOWN:
                    y_move = 0.2

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_move = 0

                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    y_move = 0
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     if event.button == 4:
            #         glTranslatef(0.0, 0.0, 1)
            #     if event.button == 5:
            #         glTranslatef(0.0, 0.0, -1)
        # glRotatef(1, 3, 1, 1)  # rotation

        x = glGetDoublev(GL_MODELVIEW_MATRIX)

        camera_x = x[3][0]
        camera_y = x[3][1]
        camera_z = x[3][2]

        # if camera_z < -1:
        #     object_passed = True

        cur_x += x_move
        cur_y += y_move

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glTranslatef(x_move, y_move, 2)

        for each_cube in cube_dict:
            Cube(cube_dict[each_cube])

        for each_cube in cube_dict:
            if camera_z <= cube_dict[each_cube][0][2]:
                # print("passed a cube")
                new_max = int(-1*(camera_z-max_distance))

                cube_dict[each_cube] = set_vertices(new_max, int(camera_z-(max_distance/2)), cur_x, cur_y)

        # Cube()
        pygame.display.flip()
        pygame.time.wait(10)

# for x in range(10):
#     main()

main()
pygame.quit()
quit()
