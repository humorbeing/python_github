from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

MyVertices = (
    (-0.25, -0.25, 0.25), (-0.25, 0.25, 0.25),
    (0.25, 0.25, 0.25), (0.25, -0.25, 0.25),
    (-0.25, -0.25, -0.25), (-0.25, 0.25, -0.25),
    (0.25, 0.25, -0.25), (0.25, -0.25, -0.25),
)
MyColors = (
    (0.2, 0.2, 0.2), (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
)
MyVertexList = (
    0, 3, 2, 1,
    2, 3, 7, 6,
    0, 4, 7, 3,
    1, 2, 6, 5,
    4, 5, 6, 7,
    0, 1, 5, 4,
)


def MyDisplay():
    global MyVertices, MyColors, MyVertexList
    glClear(GL_COLOR_BUFFER_BIT)
    glFrontFace(GL_CCW)
    glEnable(GL_CULL_FACE)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, MyColors)
    glVertexPointer(3, GL_FLOAT, 0, MyVertices)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(30.0, 1.0, 1.0, 1.0)
    for i in range(6):
        glDrawElements(GL_POLYGON, 4, GL_UNSIGNED_BYTE, MyVertexList[4*i:4*(i+1)-1])
    glFlush()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGB)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(0, 0)
    glutCreateWindow(b"OpenGL Drawing Example")  # not only string, put 'b' in front of string.
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glutDisplayFunc(MyDisplay)
    glutMainLoop()

if __name__ == '__main__':
    main()
