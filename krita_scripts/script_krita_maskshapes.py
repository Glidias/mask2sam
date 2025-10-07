import json
import os
import shutil
import tempfile
from krita import Krita
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QEventLoop, Qt, QByteArray
from PyQt5.QtWidgets import QMessageBox, QWidget, QProgressDialog, QApplication
import json, urllib.request
import uuid

# assumes you have websocket-client lib manually installed in Krita
import websocket as websocket_client

##
#  This sample script integrates Krita with ComfyUI via api_workflow/mask2sam.json workflow
#  to process the currently selected Mask Layer (and respective frame(s)) to replace any drawn shapes mask pointing shapes
#  on that layer to the full masked segmentation.
#  If doing animation, this requires a keyframe marking the shapes for the starting frame before the script runs!
#  Failure to do so will crash Krita.
# Will process only the implied current playback frame range if a range of frames is selected.
# If only 1 frame is selected from frame #1 (second frame) , will only process that single frame.
# If selecting only frame #0 (first frame), will process the defined document playback range explicitly.

# Refer to the line `# Save meta.json SETTINGS`` for hardcoded preset SAM2 and mask processing settings
#
# If this script unforeseeably crashes Krita,
# you'd probably need to manually clean up any tempdirs generated from Python prefixed with "krita_comfy_".
# Usually on Windows this is at C:\Users\<username>\AppData\Local\Temp\
# On Mac, usually at /var/folders/...
# etc. as the transfering of input/output files is done there before the script finishes and exits
# (either due to success or caught errors)
#
# After transfering input files over, the ComfyUI process runs and you can possibly continue using Krita as it won't
#.freeze the UI. After the process is done, the output mask(s) will be applied to the originally selected mask layer
#
# TODO/YAGNI: Currently only supports a single mask layer attached to a paint layer.
# Consider: Group Layer and multiple mask layers.
#
# Note: This Krita script can be binded to a suitable Ten Scripts hotkey OR run from the Scripter console directly
#
##

CLIENT_ID = str(uuid.uuid4())

def save_input_frames_with_progress(tmpdir, src_layer, active_layer, canvas_dims, start_frame, end_frame):
    """Save input frames with progress dialog (main thread only)"""
    total_frames = end_frame - start_frame + 1

    # Create progress dialog
    progress = QProgressDialog("Saving input frames before ComfyUI process...", "Cancel", 0, total_frames)
    progress.setWindowModality(Qt.WindowModal)
    progress.setWindowTitle("Input Processing")
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.show()

    try:
        # Save mask
        QApplication.processEvents()

        pixel_data = active_layer.pixelDataAtTime(0, 0, canvas_dims[0], canvas_dims[1], start_frame) if is_animated_masklayer else active_layer.pixelData(0, 0, canvas_dims[0], canvas_dims[1])
        qimg = QImage(pixel_data, canvas_dims[0], canvas_dims[1], QImage.Format_Grayscale8)

        # check if image is fully white 255
        all_white = all(qimg.pixelColor(x, y).red() == 255 for x in range(qimg.width()) for y in range(qimg.height()))
        if all_white:
            raise Exception("No mask shape regions found at current frame.")
        mask_path = os.path.join(tmpdir, "input_mask", "mask.png")
        if not qimg.save(mask_path, "PNG"):
            raise Exception("Failed to save mask")

        # Save source frames
        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            # Update progress
            progress.setValue(i)
            QApplication.processEvents()  # Keep UI responsive

            if progress.wasCanceled():
                return False
                # raise Exception("User canceled")

            # Save source frame
            pixel_data = src_layer.pixelDataAtTime(0, 0, canvas_dims[0], canvas_dims[1], frame) if is_animated_srclayer else src_layer.pixelData(0, 0, canvas_dims[0], canvas_dims[1])
            qimg = QImage(pixel_data, canvas_dims[0], canvas_dims[1], QImage.Format_ARGB32)
            img_path = os.path.join(tmpdir, "input_src", f"frame_{frame:04d}.png")
            if not qimg.save(img_path, "PNG"):
                raise Exception(f"Failed to save source frame {frame}")

        progress.setValue(total_frames)
        return True
    finally:
        progress.close()

class ComfyWorker(QThread):
    finished = pyqtSignal(str)  # emits tmpdir
    error = pyqtSignal(str)

    def __init__(self, tmpdir):
        super().__init__()
        self.tmpdir = tmpdir

    def run(self):
        try:
            # Submit prompt
            data = json.dumps({"prompt": workflow, "client_id": CLIENT_ID }).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data, headers={"Content-Type": "application/json"})
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            prompt_id = result["prompt_id"]
            client_id = CLIENT_ID
            print(f"ComfyWorker started for prompt_id: {prompt_id} and client_d: {client_id}")
            # Now connect to WebSocket
            ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
            websocket = websocket_client.create_connection(ws_url)
            try:
                websocket.recv()  # skip initial message
                while True:
                    raw = websocket.recv()
                    if isinstance(raw, str):
                        msg = json.loads(raw)
                        is_my_prompt = ("data" in msg and "prompt_id" in msg["data"] and msg["data"]["prompt_id"] == prompt_id)
                        if msg["type"] == "execution_error" and is_my_prompt:
                            error_msg = msg["data"].get("exception_message", "Unknown error")
                            raise Exception(f"ComfyUI error: {error_msg}")
                        if msg['type'] == 'executing' and is_my_prompt:
                            data = msg['data']
                            if data['node'] is None:
                                self.finished.emit(self.tmpdir)
                                return #Execution is done
                        """ Fail-safe empty queue check may be useful
                        if msg["type"] == "status":
                            if msg["data"]["status"].get("exec_info", {}).get("queue_remaining") == 0:
                                self.finished.emit(self.tmpdir)
                                return
                        """
            finally:
                websocket.close()

        except Exception as e:
            print(f"Error in ComfyWorker: {str(e)}")
            self.error.emit(str(e))

# check if __file__ is defined (it may not be in some environments)
if '__file__' not in globals():
    __file__ = os.path.abspath('C:/ComfyUI/custom_nodes/mask2sam/script_krita.py')
API_BASE_DIR = os.path.join(os.path.dirname(__file__), "api_workflows")

def find_appropriate_parent_and_target(doc, selected_node, has_masks):
    if not selected_node:
        return doc.rootNode(), None

    if has_masks:
        if selected_node.type() == "paintlayer":
            return selected_node.parentNode() or doc.rootNode(), selected_node
        if selected_node.type() == "transparencymask":
            parent_layer = selected_node.parentNode()
            if parent_layer:
                return parent_layer.parentNode() or doc.rootNode(), parent_layer
        parent = selected_node.parentNode()
        if parent and parent.type() == "grouplayer":
            return parent, None
        if selected_node.type() == "grouplayer":
            return selected_node, None
        return selected_node.parentNode() or doc.rootNode(), None
    else:
        if selected_node.type() == "grouplayer":
            return selected_node, None
        else:
            return selected_node.parentNode() or doc.rootNode(), None

def qimage_to_rgba8_bytes(qimg):
    if qimg.format() != QImage.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format_RGBA8888)

    # return ptr.asstring(qimg.byteCount()) if ptr else b''
    return QByteArray(qimg.bits().asstring(qimg.byteCount()))


def qimage_to_grayscale_bytes(qimg):
    if qimg.format() != QImage.Format_Grayscale8:
        qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
    return QByteArray(qimg.bits().asstring(qimg.byteCount()))


def logErrMessage(msg, title="Script Error"):
    QMessageBox.information(QWidget(), title, msg)

def logAndRaiseErrMessage(msg, title="Script Error"):
    logErrMessage(msg, title)
    raise Exception(msg)


# Get active document
app = Krita.instance()
doc = app.activeDocument()
if not doc:
    logAndRaiseErrMessage("❌ No active document")

canvas_dimensions = [doc.width(), doc.height()]

start_frame = doc.currentTime()
end_frame = doc.playBackEndTime()

if end_frame == start_frame:
    start_frame = doc.playBackStartTime() # highighted range back tracking
elif doc.playBackStartTime() < doc.currentTime():
    end_frame = start_frame # single frame segmentation only

active_layer = doc.activeNode()

if start_frame > end_frame:
   logAndRaiseErrMessage("❌ Invalid selected frame range in document")

if not active_layer:
    logAndRaiseErrMessage("❌ Please selecct a layer to run this script")

if active_layer.type() != "transparencymask":
    logAndRaiseErrMessage("Other layer types besides mask not ot supported atm!: " + active_layer.type())


# Setup files
# handle mask layer to get src layer
src_layer = active_layer.parentNode()
parent_layer = src_layer
if not src_layer or src_layer.type() != "paintlayer": # will support Group Layer resolution later
    logAndRaiseErrMessage("temp for now: Please select a mask layer that is attached to a paint layer")

tmpdir = tempfile.mkdtemp(prefix="krita_comfy_")
user_canceled = False

is_animated_masklayer = active_layer.animated()
is_animated_srclayer = src_layer.animated()
if not is_animated_srclayer:
    end_frame = start_frame  # force single frame if src not animated

if is_animated_masklayer and not active_layer.hasKeyframeAtTime(start_frame):
    logAndRaiseErrMessage("❌ Please make sure the selected mask layer has a keyframe at the start frame of the selected range.")

try:
    tmpdir_unix = tmpdir.replace("\\", "/")
    os.makedirs(os.path.join(tmpdir, "input_src"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "input_mask"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "output"), exist_ok=True)

    # Save frames with progress dialog
    user_canceled = not save_input_frames_with_progress(
        tmpdir, src_layer, active_layer,
        canvas_dimensions, start_frame, end_frame
    )

    # Save meta.json SETTINGS
    meta = {
        # JSONKeyCheckerNode settings in api_workflows/mask2sam.json
        "individual_objects": True, "use_bboxes": False,
        "expand": 5,
        "tapered_corners": False,

        "fill_holes": False,
        # "invert_mask": False, # likely won't need this but usable
    }
    with open(os.path.join(tmpdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Load workflow
    with open(os.path.join(API_BASE_DIR, "mask2sam.json"), "r", encoding="utf-8") as f:
        workflow_str = f.read().replace("WORKING_FOLDER_LOCATION", tmpdir_unix)
        if start_frame == end_frame: # to keep consisent as video instead?
            workflow_str = workflow_str.replace('"segmentor": "video"', '"segmentor": "single_image"')
    workflow = json.loads(workflow_str)
    print(f"✅set up temp dir: with files saved {tmpdir}", "ss")
except Exception as e:
    print("clearing up tempdir due to error before Worker:" + str(e))
    shutil.rmtree(tmpdir, ignore_errors=True)
    logAndRaiseErrMessage(f"Setup failed: {str(e)}")


# response handlers
def on_success(tmpdir):
    output_folder = os.path.join(tmpdir, "output")
    output_images = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.webp'))])
    if not output_images:
        raise Exception("No output images found")

    doc_width = doc.width()
    doc_height = doc.height()

    # insert new layer(s)
    doc.setActiveNode(parent_layer)

    # add new transpaenrcy mask
    #sel = Selection()
    #sel.select(0, 0, doc_width, doc_height, 255)
    #doc.setSelection(sel)
    #app.action('add_new_transparency_mask').trigger()
    #doc.setCurrentTime(start_frame)

    doc.setActiveNode(active_layer)

    # hack workaround to set base state of animated mask layer to white
    def reset_mask_base_to_white(mask_layer, width, height):

        white_image = QImage(width, height, QImage.Format_Grayscale8)
        white_image.fill(255)
        white_data = qimage_to_grayscale_bytes(white_image)

        original_time = doc.currentTime()

        # Try to find a frame without keyframe
        test_frame = None
        for frame in range(0, 1000):
            if not mask_layer.hasKeyframeAtTime(frame):
                test_frame = frame
                break

        if test_frame is not None:
            # Found a frame without keyframe - use it for base state
            doc.setCurrentTime(test_frame)
            mask_layer.setPixelData(white_data, 0, 0, width, height)
        else:
            # All frames have keyframes - temporarily clear one to set base state
            if mask_layer.animated():
                # Clear the first keyframe temporarily
                first_keyframe = None
                for frame in range(0, 1000):
                    if mask_layer.hasKeyframeAtTime(frame):
                        first_keyframe = frame
                        break

                if first_keyframe is not None:
                    # Store the first keyframe data
                    first_data = mask_layer.pixelDataAtTime(0, 0, width, height, first_keyframe)
                    # Clear it to access base state
                    mask_layer.setCurrentTime(first_keyframe)
                    # Unfortunately, Krita doesn't have a direct way to clear single keyframe
                    # So we'll use a different approach

                    # Alternative: Set base state on a frame outside normal range
                    doc.setCurrentTime(9999)  # Use a very high frame number
                    mask_layer.setPixelData(white_data, 0, 0, width, height)
                    # Then restore the first keyframe
                    doc.setCurrentTime(first_keyframe)
                    mask_layer.setPixelData(first_data, 0, 0, width, height)

        doc.setCurrentTime(original_time)


    def create_keyframes():
        add_blank_frame = app.action('add_blank_frame')
        # fill full white 255,255,255
        new_layer = doc.activeNode()

        # confirm new_layer is transparencymask
        if new_layer.type() != "transparencymask":
            # logAndRaiseErrMessage("Failed to create new transparency mask layer:" + new_layer.type())
            active_window = app.activeWindow()
            active_view = active_window.activeView() if active_window else None
            if active_view:
                active_view.showFloatingMessage("Failed to create new transparency mask layer:" + new_layer.type(),  app.icon("16_light_info"), 4000, 1)
            return

        #white_image = QImage(doc_width, doc_height, QImage.Format_Grayscale8)
        # white_image.fill(255)
        #white_data = qimage_to_grayscale_bytes(white_image)
        original_time = doc.currentTime()
        #doc.setCurrentTime(0)
        #new_layer.setPixelData(white_data, 0, 0, doc_width, doc_height)

        reset_mask_base_to_white(new_layer, doc_width, doc_height)

        for i, img_name in enumerate(output_images):
            frame_num = start_frame + i
            doc.setCurrentTime(frame_num)
            add_blank_frame.trigger()

         # for i, img_name in enumerate(output_images):
         #   frame_num = start_frame + i
         #   doc.setCurrentTime(frame_num)
          #  new_layer.setPixelData(white_data, 0, 0, white_image.width(), white_image.height())

        for i, img_name in enumerate(output_images):
            frame_num = start_frame + i
            doc.setCurrentTime(frame_num)
            img_path = os.path.join(output_folder, img_name)
            qimg = QImage(img_path)
            pixel_data = qimage_to_grayscale_bytes(qimg)
            new_layer.setPixelData(pixel_data, 0, 0, qimg.width(), qimg.height())

        doc.setCurrentTime(original_time)

        doc.refreshProjection()

    if active_layer.animated():
        QTimer.singleShot(0, create_keyframes)
    else:
        img_path = os.path.join(output_folder, output_images[0])
        qimg = QImage(img_path)
        pixel_data = qimage_to_grayscale_bytes(qimg)
        active_layer.setPixelData(pixel_data, 0, 0, qimg.width(), qimg.height())
        doc.refreshProjection()

def on_error(msg):
    print("clearing up tempdir due to error in Worker")
    shutil.rmtree(tmpdir, ignore_errors=True)
    logErrMessage(f"ComfyUI Error: {msg}")

# Start worker ?
try:
    active_window = app.activeWindow()
    active_view = active_window.activeView() if active_window else None
    if not user_canceled:
        # logErrMessage("Starting ComfyUI process, please wait... " + str(tmpdir))
        worker = ComfyWorker(tmpdir)
        worker.finished.connect(on_success)
        worker.error.connect(on_error)
        worker.start()  # Non-blocking!
        if active_view:
            active_view.showFloatingMessage("ComfyUI process has started...", app.icon("16_light_info"), 4000, 1)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.error.connect(loop.quit)
        loop.exec_()
    else:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if active_view:
            active_view.showFloatingMessage("Canceled script",  app.icon("16_light_info"), 1000, 1)
except Exception as e:
    print("clearing up tempdir due to error starting Worker:" + str(e))
    shutil.rmtree(tmpdir, ignore_errors=True)
