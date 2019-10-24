import sys
import numpy
import cv2
import argparse
import time

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'

try:
    from openvino.inference_engine import IENetwork, IEPlugin
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)

ARGS = None
TEXT_SIZE = 0.5
LABEL_BG_COLOR = (75, 75, 75)
TEXT_COLOR = (0, 255, 0)
gNetworkMean= None
gNetworkStd = None
CLASSIFIER_WINDOW_NAME = 'clean_water_ai_by_vinu'

def build_parser():
    parser = argparse.ArgumentParser(description = 'Image classifier using \
                         Intel® Movidius™ Neural Compute Stick.' )
    parser.add_argument( '--ir', metavar = 'IR_FILE',
                        type=str, default = '../../caffe/SqueezeNet/squeezenet_v1.0.xml',
                         help = 'Absolute path to the neural network IR file. Default = ../../caffe/SqueezeNet/squeezenet_v1.0.xml.')
    parser.add_argument( '-l', '--labels', metavar = 'LABELS_FILE',
                        type=str, default = '../../data/ilsvrc12/synset_labels.txt',
                         help='Absolute path to labels file. Default = ../../data/ilsvrc12/synset_labels.txt.')
    parser.add_argument("-ld", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    parser.add_argument( '-stat', '--stat', metavar = 'STAT_FILE',
                        type=str, default = None,
                         help = 'Stat file. Default = None.')
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
    print('   - ' + YELLOW + 'Labels File:  ' + NOCOLOR, ARGS.labels)
    print('   - ' + YELLOW + 'Stat File:    ' + NOCOLOR, ARGS.stat)
    print('')
    print(' Press any key to exit.')
    print('')

def getMeanValue():
    # load means and stds from stat.txt
    with open(ARGS.stat, 'r') as f:
        gNetworkMean = f.readline().split()
        gNetworkStd = f.readline().split()
        for i in range(3):
            gNetworkMean[i] = 255 * float(gNetworkMean[i])
            gNetworkStd[i] = 1 / (255.0 * float(gNetworkStd[i]))
            return gNetworkMean, gNetworkStd

def gNetworkCategories():
    # Load categories from categories.txt
    gNetworkCategories = []
    with open(ARGS.labels, 'r') as f:
        for line in f:
            cat = line.split('\n')[0]
            if cat != 'classes':
                gNetworkCategories.append(cat)
        f.close()
    #print('Number of categories:', len(gNetworkCategories))
    return gNetworkCategories

def preprocess_image(img,input_shape):
    global ARGS
    # get input shapes
    n = input_shape[0]
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    
    #buffer_data_type = numpy.dtype(numpy.uint8) # the buffer contains 8 bit unsigned ints that are the RGB Values of the image
    #image_unit8_array = numpy.frombuffer(img, buffer_data_type, -1, 0) # get the input image into an array
    #actual_stream_width = int(round((2*w+1)/2)) # hack, rather get this from the app sink
    #image_unit8_array = image_unit8_array.reshape(w,h,1)
    #image_unit8_array = image_unit8_array[0:(h-1),0:(w-1),0:3]    # crop to network input size
    #image_float_array = image_unit8_array.astype('float32')
    #Image preprocessing
    #img = numpy.array(img).astype(numpy.uint8)
    img = cv2.flip(img, 1)
    img = cv2.resize(img, tuple((h, w)))
    img = img.astype(numpy.float32)
    gNetworkMean,gNetworkStd =  getMeanValue()
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - float(gNetworkMean[i])) * float(gNetworkStd[i])
    transposed_img = numpy.transpose(img, (2,0,1))
    reshaped_img = transposed_img.reshape((n, c, h, w))
    return reshaped_img.astype(numpy.float16)

def perform_inference(img, exec_net, input_blob, output_blob):
    # Prepare Categories for age and gender networks
    with open(ARGS.labels) as labels_file:
        #label_list = labels_file.read().splitlines()
        label_list=gNetworkCategories()

    start_time = time.time()
    # Run async inference
    req_handle = exec_net.start_async(request_id=0, inputs={input_blob: img})
    # Get results from async inference
    status = req_handle.wait()
    results = req_handle.outputs[output_blob]
    end_time = time.time()
    
    #Process the results
    for i, probs in enumerate(results):
        probs = numpy.squeeze(probs)
        top_ind = numpy.argsort(probs)[-1:][::-1]
        
    predicted_label = label_list[int(top_ind[0])]
    predicted_confidence = '{0:3.1f}'.format(float(100*results[0][top_ind]))
    return predicted_label, predicted_confidence,end_time - start_time

def output_processing(result_label, result_confidence, WIN_W, WIN_H):
    
        # Form the text to display in our window
    text_to_display = result_label + ' - ' + str(result_confidence) + '%'
    # Get text coordinates
    text_size = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, 1)[0]
    text_coord_w = int( (WIN_W - text_size[0]) / 2 )
    text_coord_h = WIN_H - text_size[1]
    # Get box coordinates
    box_left_top_w = 0
    box_left_top_h = WIN_H - text_size[1]*3
    
    return text_to_display, (text_coord_w, text_coord_h), (box_left_top_w, box_left_top_h)

####################### Entrypoint for the application #######################
def main():
    global ARGS
    ARGS = build_parser().parse_args()

    # Set up the network and plugin
    input_shape, input_blob, output_blob, exec_net = setup_network()
    # Set the camera capture properties
    cap = cv2.VideoCapture(ARGS.source)
    CAM_W = ARGS.cap_res[0]
    CAM_H = ARGS.cap_res[1]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H);

    # Set the window properties
    WIN_W =  ARGS.win_size[0]
    WIN_H =  ARGS.win_size[1]
    cv2.namedWindow(CLASSIFIER_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CLASSIFIER_WINDOW_NAME, WIN_W, WIN_H)

    elapsed_time = 0
    frame_count = 0
    fps = 0

    while (True):
        # Read image from camera, get camera width and height
        ret, img = cap.read()
        frame_count += 1
        # Preprocess the image
        input_img = preprocess_image(img,input_shape)

        # Perform the inference and get the results
        res_label, res_confidence, e_time = perform_inference(input_img, exec_net, input_blob, output_blob)

        elapsed_time = elapsed_time + e_time
        fps = frame_count / elapsed_time

        text,text_coords, box_coords = output_processing(res_label, res_confidence, WIN_W, WIN_H)

         # Display the image. Quit if user presses q.
        cv2.rectangle(img, box_coords, (WIN_W, WIN_H), LABEL_BG_COLOR, -1)
        cv2.putText(img, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 1)

        cv2.imshow(CLASSIFIER_WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key != -1:
        	break

    print(" Frame per second: {0:.2f}".format(fps))
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(' Finished.')

if __name__ == '__main__':
    sys.exit(main())