from pynput.keyboard import Key, Controller
import time
keyboard = Controller()
time.sleep(10)
# keyboard.press('w')
# time.sleep(2)
# keyboard.release('w')
# keyboard.type('abc')
while True:
    time.sleep(2)
    keyboard.press('a')
    time.sleep(2)
    keyboard.release('a')
    time.sleep(2)
    keyboard.press('d')
    time.sleep(2)
    keyboard.release('d')
    time.sleep(2)
    keyboard.press(Key.space)
    time.sleep(2)
    keyboard.release(Key.space)

# from pynput import keyboard
#
# def on_press(key):
#     try:
#         print('alphanumeric key {0} pressed'.format(
#             key.char))
#     except AttributeError:
#         print('special key {0} pressed'.format(
#             key))
#
# def on_release(key):
#     print('{0} released'.format(
#         key))
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False
#
# # Collect events until released
# with keyboard.Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()
#
# # ...or, in a non-blocking fashion:
# listener = keyboard.Listener(
#     on_press=on_press,
#     on_release=on_release)
# listener.start()
# dsbsdfsdfsdfsdfsfsdfsssws ws ws ws ws w