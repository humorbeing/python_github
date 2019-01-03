from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
WINDOW_POSITION_X = 0
WINDOW_POSITION_Y = 0
R = 1.5
P = 0.2
D = 0.17

Rm = 3
Dm = 4.5

planets = {
    'Mercury': {'radius': 0.38,
                'period': 0.24,
                'angle':  0.0,
                'distance': 57.9,
                'moons': {},
                },
    'Venus':   {'radius': 0.95,
                'period': 0.62,
                'angle':  0.0,
                'distance': 108.2,
                'moons': {},
                },
    'Earth':   {'radius': 1,
                'period': 1,
                'angle': 0.0,
                'distance': 149.6,
                'moons': {
                    '1': {
                        'radius': 0.272,
                        'period': 0.074,
                        'angle': 0.0,
                        'distance': 0.386+1,
                    }
                },
                },
    'Mars':    {'radius': 0.53,
                'period': 1.9,
                'angle': 0.0,
                'distance': 227.9,
                'moons': {
                    '1': {
                        'radius': 0.272,
                        'period': 0.074,
                        'angle': 0.0,
                        'distance': 0.386+0.53,
                    },
                    '2': {
                        'radius': 0.272,
                        'period': 0.144,
                        'angle': 0.0,
                        'distance': 0.786+0.53,
                    }
                },
                },
    'Jupiter': {'radius': 11.2,
                'period': 11.9,
                'angle': 0.0,
                'distance': 778.4,
                'moons': {

                },
                },
    'Saturn':  {'radius': 9.5,
                'period': 29.4,
                'angle': 0.0,
                'distance': 1426.7,
                'moons': {

                },
                }
}


def drawMoon(distance, angle, planetRadius):
    glColor3f(0, 0.5, 1)
    glPushMatrix()
    glBegin(GL_LINE_STRIP)
    for i in range(0, 361):
        theta = 2.0 * 3.141592 * i / 360.0
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        glVertex3f(x, 0, y)
    glEnd()
    glColor3f(1, 0, 1)
    glRotatef(angle, 0, 1, 0)
    glTranslatef(distance, 0, 0)
    glutWireSphere(planetRadius, 20, 20)
    glPopMatrix()


def drawPlanet(distance, angle, planetRadius, moons):
    glColor3f(0, 0.5, 1)
    glPushMatrix()
    glBegin(GL_LINE_STRIP)
    for i in range(0, 361):
        theta = 2.0 * 3.141592 * i / 360.0
        x = distance * math.cos(theta)
        y = distance * math.sin(theta)
        glVertex3f(x, 0, y)
    glEnd()
    glColor3f(0, 1, 1)
    glRotatef(angle, 0, 1, 0)
    glTranslatef(distance, 0, 0)
    glutWireSphere(planetRadius, 20, 20)
    for moon in moons:
        moons[moon]['angle'] += (1 / moons[moon]['period']) * P
        drawMoon(moons[moon]['distance'] * Dm,
                 moons[moon]['angle'],
                 moons[moon]['radius'] * Rm)
    glPopMatrix()


def drawScene():
    global planets

    glColor3f(1, 1, 0)
    glutSolidSphere(3, 20, 20)

    for planet in planets:
        planets[planet]['angle'] += (1 / planets[planet]['period']) * P
        drawPlanet(planets[planet]['distance'] * D,
                   planets[planet]['angle'],
                   planets[planet]['radius'] * R,
                   planets[planet]['moons'])


def disp():
    glClear(GL_COLOR_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30, 1.0, 0.1, 1000)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glViewport(WINDOW_POSITION_X, WINDOW_POSITION_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
    gluLookAt(0, 1000, 0, 0, 0, 0, 0, 0, 1)
    drawScene()

    glFlush()


# windowing
glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
glutInitWindowPosition(WINDOW_POSITION_X, WINDOW_POSITION_Y)
glutCreateWindow(b"Simple Solar")

glClearColor(0, 0.0, 0.0, 0)

# register callbacks
glutDisplayFunc(disp)
glutIdleFunc(disp)

# enter main infinite-loop
glutMainLoop()

