import argparse
import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gdk', '3.0')
gi.require_version('Gtk', '3.0')
gi.require_version('GLib','2.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gdk
from gi.repository import Gst
from gi.repository import Gtk
from gi.repository import GstVideo
from gi.repository import GLib
from gi.repository import GdkX11

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from queue import Queue
from gi.repository import GLib
from threading import Thread

from gi.repository import Gst

import numpy


NETWORK_IMAGE_WIDTH = 227           # the width of images the network requires
NETWORK_IMAGE_HEIGHT = 227          # the height of images the network requires
NETWORK_IMAGE_FORMAT = "RGB"
GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
GST_APP_NAME = "app"            # gstreamer sink name
GST_VIEW_NAME = "view"          # gstreamer view sink name
CAMERA_INDEX = "0"              # 0 is first usb cam, 1 the second etc.
SINK_NAME="xvimagesink"         # use for x86-64 platforms
#SINK_NAME="glimagesink"	# use for Raspian Jessie platforms

# Globals for the program
gGstAppSink = None
gIt = None
gRunning = False
gOt = None
gNetworkMean = None
gNetworkStd = None
gNetworkCategories = None
gUpdateq = Queue()
gGraph = None
gCallback = None
gResultLabel = Gtk.Label()          # label to display inferences in
gDrawAreaSink = Gtk.DrawingArea()   # DrawingArea to display camera feed in.
# end of globals for the program
# Ref -- https://github.com/movidius/ncappzoo/blob/openvino_2019_r3/apps/simple_classifier_py_camera/run.py

try:
    from openvino.inference_engine import IENetwork, IEPlugin
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

ARGS = None

def build_parser():
    parser = argparse.ArgumentParser(description = 'Clean water ai using \
                         Intel® Movidius™ Neural Compute Stick 2.' )
    parser.add_argument( '--ir', metavar = 'IR_FILE',
                        type=str, default = '../../cleanwaterai/graph.xml',
                         help = 'Absolute path to the neural network IR file. Default = ../../cleanwaterai/graph.xml.')
    parser.add_argument( '-stat', '--statFile', metavar = 'STAT_FILE',
                        type=str, default = '../../cleanwaterai/stat.txt',
                         help='Absolute path to labels file. Default = ../../cleanwaterai/stat.txt')
    parser.add_argument( '-category', '--category', metavar = 'CATEGORY_FILE',
                        type=str, default = '../../cleanwaterai/category.txt',
                         help='Absolute path to labels file. Default = ../../cleanwaterai/category.txt')
    parser.add_argument( '-s', '--source', metavar = 'CAMERA_SOURCE',
                        type=int, default = 0, help = 'V4L2 Camera source. Default = 0.')
    parser.add_argument( '-c', '--cap_res', metavar = 'CAMERA_CAPTURE_RESOLUTION',
                        type=int, default = (1280, 960), help = 'Camera capture resolution. Default = (1280, 960).')
    parser.add_argument( '-w', '--win_size', metavar = 'WINDOW SIZE',
                        type=int, default = (640, 480), help = 'Inference result window size. Default = (640, 480).')

    return parser


    def setup_network():
    global ARGS
    # Select the myriad plugin and IRs to be used
    plugin = IEPlugin(device='MYRIAD')
    net = IENetwork(model = ARGS.ir, weights = ARGS.ir[:-3] + 'bin')
    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    # Load the network and get the network shape information
    exec_net = plugin.load(network = net)
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    del net
    display_info(input_shape, output_shape)

    return input_shape, input_blob, output_blob, exec_net

    def display_info(input_shape, output_shape):
    print()
    print(YELLOW + 'Starting application...' + NOCOLOR)
    print('   - ' + YELLOW + 'Camera Source:' + NOCOLOR, ARGS.source)
    print('   - ' + YELLOW + 'Plugin:       ' + NOCOLOR + 'Myriad')
    print('   - ' + YELLOW + 'IR File:      ' + NOCOLOR, ARGS.ir)
    print('   - ' + YELLOW + 'Input Shape:  ' + NOCOLOR, input_shape)
    print('   - ' + YELLOW + 'Output Shape: ' + NOCOLOR, output_shape)
    print('   - ' + YELLOW + 'Stats File:  ' + NOCOLOR, ARGS.statFile)
    print('   - ' + YELLOW + 'Category File:    ' + NOCOLOR, ARGS.category)
    print('')
    print(' Press any key to exit.')
    print('')

    ####################### Entrypoint for the application #######################
    def main():

        Gdk.init([])
        Gtk.init([])
        Gst.init([])

        global ARGS
        ARGS = build_parser().parse_args()

    # load means and stds from stat.txt
    with open(ARGS.statFile, 'r') as f:
        gNetworkMean = f.readline().split()
        gNetworkStd = f.readline().split()
        for i in range(3):
            gNetworkMean[i] = 255 * float(gNetworkMean[i])
            gNetworkStd[i] = 1 / (255 * float(gNetworkStd[i]))


    # Load categories from categories.txt
    gNetworkCategories = []
    with open(ARGS.category, 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                gNetworkCategories.append(cat)
        f.close()
    print('Number of categories:', len(gNetworkCategories))

    # the camera source string for USB cameras.  They will be /dev/video0, /dev/video1, etc.
    # for this sample we will open the first camera (/dev/video0)
    cam_src_str = "v4l2src device=/dev/video" + ARGS.source

    app_launch_str = "\
        videoscale ! video/x-raw, width=%s, height=%s ! \
        videoconvert ! video/x-raw, format=%s ! \
        appsink name=%s max-buffers=1 drop=true enable-last-sample=true" % (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT, NETWORK_IMAGE_FORMAT, GST_APP_NAME )


        # build GUI
    window = Gtk.Window()
    window.connect("delete-event", window_closed, gstPipeline)
    WINDOW_WIDTH = ARGS.win_size[0]
    WINDOW_HEIGHT = ARGS.win_size[1]
    window.set_default_size (WINDOW_WIDTH, WINDOW_HEIGHT)
    window.set_title ("CLEAN WATER AI")

    box = Gtk.Box()
    box.set_spacing(5)
    box.set_orientation(Gtk.Orientation.VERTICAL)
    window.add(box)

    box.pack_start(gDrawAreaSink, True, True, 0)
    gResultLabel = Gtk.Label()

    box.pack_start(gResultLabel, False, True, 0)

    window.show_all()
    window.realize()
    gstPipeline.get_by_name(GST_VIEW_NAME).set_window_handle(gDrawAreaSink.get_window().get_xid())

    # Set up the network and plugin
    input_shape, input_blob, output_blob, exec_net = setup_network()

    # close the device
    del exec_net
    del plugin
    if __name__ == '__main__':
        sys.exit(main() or 0)
