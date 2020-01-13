from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

FlatShaded = 0
Wireframed = 0
ViewX = 0
ViewY = 0


def InitLight():
    mat_diffuse = (0.5, 0.4, 0.3, 1.0)
    mat_specular = (1.0, 1.0, 1.0, 1.0)
    mat_ambient = (0.5, 0.4, 0.3, 1.0)
    mat_shininess = (15.0)
    light_specular = (1.0, 1.0, 1.0, 1.0)
    light_diffuse = (0.8, 0.8, 0.8, 1.0)
    light_ambient = (0.3, 0.3, 0.3, 1.0)
    light_position = (-3, 6, 3.0, 0.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)


def MyMouseMove(X, Y):
    glutPostRedisplay()


def MyKeyboard(*ker, x, y):
    pass


def MyDisplay():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0)
    glutSolidTeapot(0.2)
    glFlush()


def MyReshape(w, h):
    glViewport(0, 0, GLsizei(w), GLsizei(h))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(400, 400)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Sample Drawing")  # not only string, put 'b' in front of string.
    glClearColor(0.4, 0.4, 0.4, 0.0)
    InitLight()
    glutDisplayFunc(MyDisplay)
    glutKeyboardFunc(MyKeyboard)
    glutMouseFunc(MyMouseMove)
    glutReshapeFunc(MyReshape)
    glutMainLoop()

if __name__ == '__main__':
    main()