#!/usr/bin/python3
# Python script to start a USB camera and feed frames to
# the Movidius Neural Compute Stick that is loaded with a
# CNN graph file and report the inferred results

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
import argparse

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
SINK_NAME = "xvimagesink"         # use for x86-64 platforms
# SINK_NAME="glimagesink"	# use for Raspian Jessie platforms

# Globals for the program
gGstAppSink = None
gIt = None
gRunning = False
gOt = None
gNetworkMean = None
gNetworkStd = None
gNetworkCategories = None
gUpdateq = Queue()
gCallback = None
gResultLabel = Gtk.Label()          # label to display inferences in
gDrawAreaSink = Gtk.DrawingArea()   # DrawingArea to display camera feed in.
# end of globals for the program
# Ref -- https://github.com/movidius/ncappzoo/blob/openvino_2019_r3/apps/simple_classifier_py_camera/run.py

try:
    from openvino.inference_engine import IENetwork, IEPlugin
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW +
          ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

ARGS = None

def build_parser():
    parser = argparse.ArgumentParser(description='Clean water ai using \
                         Intel® Movidius™ Neural Compute Stick 2.')
    parser.add_argument('--ir', metavar='IR_FILE',
                        type=str, default='../../cleanwaterai/graph.xml',
                        help='Absolute path to the neural network IR file. Default = ../../cleanwaterai/graph.xml.')
    parser.add_argument('-stat', '--statFile', metavar='STAT_FILE',
                        type=str, default='../../cleanwaterai/stat.txt',
                        help='Absolute path to labels file. Default = ../../cleanwaterai/stat.txt')
    parser.add_argument('-category', '--category', metavar='CATEGORY_FILE',
                        type=str, default='../../cleanwaterai/category.txt',
                        help='Absolute path to labels file. Default = ../../cleanwaterai/category.txt')
    parser.add_argument('-s', '--source', metavar='CAMERA_SOURCE',
                        type=int, default=0, help='V4L2 Camera source. Default = 0.')
    parser.add_argument('-c', '--cap_res', metavar='CAMERA_CAPTURE_RESOLUTION',
                        type=int, default=(1280, 960), help='Camera capture resolution. Default = (1280, 960).')
    parser.add_argument('-w', '--win_size', metavar='WINDOW SIZE',
                        type=int, default=(640, 480), help='Inference result window size. Default = (640, 480).')

    return parser

def setup_network():
    global ARGS
    # Select the myriad plugin and IRs to be used
    plugin = IEPlugin(device='MYRIAD')
    net = IENetwork(model=ARGS.ir, weights=ARGS.ir[:-3] + 'bin')
    # Set up the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    # Load the network and get the network shape information
    exec_net = plugin.load(network=net)
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    del net
    display_info(input_shape, output_shape)
    return plugin,input_shape, input_blob, output_blob, exec_net

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
    
# Start the input and output worker threads for the application


def start_thread(exec_net,input_blob, output_blob):
    """ start threads and idle handler (update_ui) for callback dispatching
    """
    global gIt, gOt, gRunning
    gRunning = True
    # TODO: inefficient, find a thread safe signal/event posting method
    GLib.idle_add(update_ui)
    gIt = Thread(target=input_thread(exec_net,input_blob, output_blob))
    gIt.start()
    gOt = Thread(target=output_thread(exec_net,input_blob, output_blob))
    gOt.start()

# Stop worker threads for the application.  Blocks until threads are terminated

def stop_thread():
    """ stop threads
    """
    global gIt, gOt, gRunning

    # Set gRunning flag to false so worker threads know to terminate
    gRunning = False

    # Wait for worker threads to terminate.
    gIt.join()
    gOt.join()
    
# Worker thread function for input to MVNC.
# Gets a preprocessed camera sample and calls the MVNC API to do an inference on the image.
def input_thread(exec_net,input_blob, output_blob):
    """ input thread function
    """
    global gRunning
    frame_number = 0
    while gRunning:
        preprocessed_image_buf = get_sample()
        start_time = time()
        if preprocessed_image_buf is not None:                                    # TODO: eliminate busy looping before samples are available
            #print("loading %s : %s" % (preprocessed_image_buf.shape, preprocessed_image_buf ))
             req_handle = exec_net.start_async(request_id=frame_number, inputs={input_blob: preprocessed_image_buf})
            #gGraph.LoadTensor(preprocessed_image_buf ,"frame %s" % frame_number)
        frame_number=frame_number+1

    print("Input thread terminating.")

# Worker thread function to handle inference results from the MVNC stick
def output_thread(exec_net,input_blob, output_blob):
  """ output thread function
  for getting inference results from Movidius NCS
  running graph specific post processing of inference result
  queuing the results for main thread callbacks
  """
  global gRunning
  cur_request_id=0
  try:
    while gRunning:
      try:
          if exec_net.requests[cur_request_id].wait(-1) == 0:
            output,user_data = exec_net.requests[cur_request_id].outputs[output_blob]
        #inference_result, user_data = gGraph.GetResult()
            cur_request_id=cur_request_id+1
            gUpdateq.put((postprocess(output), user_data))
      except KeyError:
        # This error occurs when GetResult can't access the user param from the graph, we're just ignoring it for now
        #print("KeyError")
        pass
  except Exception as e:
    print(e)
    pass
  print("Output thread terminating")


def update_ui():
    """
    Dispatch callbacks with post processed inference results
        in the main thread context
    :return: running global status
    """
    global gRunning

    while not gUpdateq.empty():
        # get item from update queue
        (out, cookie) = gUpdateq.get()
        gCallback(cookie, out)
    return gRunning

# Get a sample from the camera and preprocess it so that its ready for
# to be sent to the MVNC stick to run an inference on it.
def get_sample():
    """ get a preprocessed frame to be pushed to the graph
    """
    sample = gGstAppSink.get_property('last-sample')
    if sample:
        # a sample was available from the camera via the gstreamer app sink
        buf = sample.get_buffer()
        result, info = buf.map(Gst.MapFlags.READ)
        preprocessed_image_buffer = preprocess(info.data)
        buf.unmap(info)
        del buf
        del sample
        return preprocessed_image_buffer
    return None

# preprocess the camera images to create images that are suitable for the
# network.  Specifically resize to appropriate height and width
# and make sure the image format is correct.  This is called by the input worker
# thread function prior to passing the image the MVNC API.
def preprocess(data):
    """ preprocess a video frame
    input - in the format specified by rawinputformat() method
    output - in the format required by the graph
    """
    resize_width = NETWORK_IMAGE_WIDTH
    resize_height = NETWORK_IMAGE_HEIGHT

    buffer_data_type = numpy.dtype(numpy.uint8) # the buffer contains 8 bit unsigned ints that are the RGB Values of the image
    image_unit8_array = numpy.frombuffer(data, buffer_data_type, -1, 0) # get the input image into an array
    actual_stream_width = int(round((2*resize_width+1)/2)) # hack, rather get this from the app sink
    image_unit8_array = image_unit8_array.reshape(actual_stream_width,resize_height,3)
    image_unit8_array = image_unit8_array[0:(resize_height-1),0:(resize_width-1),0:3]    # crop to network input size
    image_float_array = image_unit8_array.astype('float32')

    #Preprocess image changing the RGB pixel values to the values the network needs
    # to do this we subtract the mean and multiply the std for each channel (R, G and B)
    # these mean and std values come from the stat.txt file that must accompany the
    # graph file for the network.
    for i in range(3):
        image_float_array[:,:,i] = (image_float_array[:,:,i] - gNetworkMean[i]) * gNetworkStd[i]

    # Finally we return the values as Float16 rather than Float32 as that is what the network expects.
    return image_float_array.astype(numpy.float16)

# post process the results from MVNC API to create a human
# readable string.
def postprocess(output):
    """ postprocess an inference result
    input - in the format produced by the graph
    output - in a human readable format
    """
    order = output.argsort()
    last = len(gNetworkCategories)-1
    text = gNetworkCategories[order[last-0]] + ' (' + '{0:.2f}'.format(output[order[last-0]]*100) + '%) '

    # to get top 5 use this code
    #for i in range(0, min(5, last+1)):
    #    text += gNetworkCategories[order[last-i]] + ' (' + '{0:.2f}'.format(output[order[last-i]]*100) + '%) '

    return text

def put_output(userobj, out):
    """ Method for receiving the (postprocessed) results
    userobj - user object passed to the FathomExpress
    out - output
    """
    global gResultLabel
    global gDrawAreaSink

    gResultLabel.set_text("%s\n" % out)
    
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
        appsink name=%s max-buffers=1 drop=true enable-last-sample=true" % (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT, NETWORK_IMAGE_FORMAT, GST_APP_NAME)

    # build GUI
        window = Gtk.Window()
        window.connect("delete-event", window_closed, gstPipeline)
        WINDOW_WIDTH = ARGS.win_size[0]
        WINDOW_HEIGHT = ARGS.win_size[1]
        window.set_default_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        window.set_title("CLEAN WATER AI")
        
        box = Gtk.Box()
        box.set_spacing(5)
        box.set_orientation(Gtk.Orientation.VERTICAL)
        window.add(box)
        
        box.pack_start(gDrawAreaSink, True, True, 0)
        gResultLabel = Gtk.Label()

        box.pack_start(gResultLabel, False, True, 0)

        window.show_all()
        window.realize()
        gstPipeline.get_by_name(GST_VIEW_NAME).set_window_handle(
        gDrawAreaSink.get_window().get_xid())

    # Set up the network and plugin
        plugin, input_shape, input_blob, output_blob, exec_net = setup_network()
        
        #Initialize input and output threads to pass images to the
    # MVNC device and to read results from the inferences made on thos images.

        gCallback = put_output
        start_thread(exec_net,input_blob, output_blob)
        
        if gstPipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            gstPipeline.set_state(Gst.State.NULL)
        else:
        # export GST_DEBUG_DUMP_DOT_DIR=/tmp/
            Gst.debug_bin_to_dot_file(
                gstPipeline, Gst.DebugGraphDetails.ALL, 'playing-pipeline')
        Gtk.main()
        Gst.debug_bin_to_dot_file(
            gstPipeline, Gst.DebugGraphDetails.ALL, 'shutting-down-pipeline')
        gstPipeline.set_state(Gst.State.NULL)
        print("exiting main loop")
        
        # close the device
        del exec_net
        del plugin
        if __name__ == '__main__':
            sys.exit(main() or 0)

