# # -*- coding: utf-8 -*-
# """
# Created on 1st May 2024

# @author: Marta, from grad_cam_saveout.py by Henry
# """

import os
# import onnx
import numpy as np
import monai.transforms
from monai.networks.nets import DenseNet201
import torch
import argparse
import scipy.ndimage as scimage
# from cropimage_MV import detect_roi
import matplotlib.pyplot as plt
from detect_roi_henry import RegionOfInterest
from medcam import medcam
# from resize import Resize
import ismrmrd
import io

show_output = True

# def parse_h5_into_fire_arguments(h5_path: str) -> Tuple[iter, str | None, ismrmrd.xsd.ismrmrdHeader]:
def parse_h5_into_fire_arguments(h5_path):
    """
    Takes an ISMRMRD h5 file and returns an iterator behaving as a Connection, a 'config` string and the metadata.

    :param h5_path:  Path to the ISMRMRD h5 file
    :return:  A tuple of:
        connection_iter:  An iterator behaving as a Connection
        config:  A string representing the config ('amp')
        metadata:  The metadata
    """
    # print(h5_path)
    ds = ismrmrd.Dataset(h5_path, '/dataset', False)

    # Images
    try:
        n_images = ds.number_of_images('images_0')
        images = [ds.read_image('images_0', i) for i in range(n_images)]
    except LookupError:
        n_images = ds.number_of_images('image_0')
        images = [ds.read_image('image_0', i) for i in range(n_images)]
    connection_iter = iter(images)
    # print(images[-1].meta)

    # XML
    metadata = ismrmrd.xsd.CreateFromDocument(ds.read_xml_header())
    # Config
    config = None

    return connection_iter, config, metadata, ds

def detect_roi(image, plot_debug):

    roi_algorithm = RegionOfInterest()
    
    # image = Resize.resample_image(image)
    # rescaled_affine = nib.affines.rescale_affine(affine, image_shape, (1,1,1))
    # rescaled_header = header
    # rescaled_header['pixdim'][1:4] = [1, 1, 1]

    # if seq != 'cine_2ch_3D':
    box = roi_algorithm.detect_roi_dilate_n_crop(image, debug=plot_debug)
    buffer = 10
    # else:
    #     box = roi_algorithm.detect_roi_dilate_n_crop_2ch(image,
    #                                                      debug=plot_debug)
    
    # image = sitk.GetArrayFromImage(image)
    # image = np.swapaxes(image, 0, -1)
    
    # For Kavitha data ONLY
    # Check if the image has a z-dimension larger than 1
    # If so, select every n-1 slice to construct a new volume to reduce the depth
    # new_img_select_temp = []
    # if int(affine[2][2]) > 4:
    #     for i in range(0, image.shape[-1], int(affine[2][2]) - 1):
    #         new_img_select_temp.append(i)
    #     image = image[:,:,(new_img_select_temp)]
    
    box_after_buffer = [box[0] - buffer, box[1] + buffer, box[2] - buffer, box[3] + buffer]
    if box_after_buffer[0] < 0:
        box_after_buffer[0] = 0
    if box_after_buffer[2] < 0:
        box_after_buffer[2] = 0
    
    cropped_image = image[box_after_buffer[2]:box_after_buffer[3], box_after_buffer[0]:box_after_buffer[1]]
    cropped_image_row = cropped_image.shape[0]
    cropped_image_col = cropped_image.shape[1]
    dimension_diff = abs(cropped_image_row - cropped_image_col)
    pad_before = dimension_diff // 2
    pad_after = dimension_diff - pad_before
    if cropped_image_row > cropped_image_col:
        cropped_image_square = np.pad(cropped_image, ((0, 0), (pad_before, pad_after), (0, 0)))
    else:
        cropped_image_square = np.pad(cropped_image, ((pad_before, pad_after), (0, 0), (0, 0)))

    np.save('CINE3ch_Cropped.npy', cropped_image_square)
    # cropped_image_square_shape = cropped_image_square.shape
    # cropped_sqaure_header = rescaled_header
    # cropped_sqaure_header['dim'][1:4] = cropped_image_square_shape
    # cropped_sqaure_affine = rescaled_affine
    # cropped_sqaure_affine = nib.affines.rescale_affine(rescaled_affine, cropped_image_square_shape, (1,1,1))
                        
    # cropped_image_square_nifti = nib.Nifti1Image(cropped_image_square,
    #                                               header = cropped_sqaure_header,
    #                                               affine = cropped_sqaure_affine)
    # file_name_3D = path.split('\\')[-1]
    # os.makedirs(FolderPath_3D, exist_ok=True)
    # nib.save(cropped_image_square_nifti, os.path.join(FolderPath_3D, file_name_3D))
    
    # cropped_sqaure_header_2D = cropped_sqaure_header
    # cropped_sqaure_header_2D['dim'][3] = 1
    # cropped_image_square_shape_2D = cropped_sqaure_header_2D['dim'][1:4]
    # cropped_sqaure_affine_2D = cropped_sqaure_affine
    # cropped_sqaure_affine_2D = nib.affines.rescale_affine(cropped_sqaure_affine, cropped_image_square_shape_2D, (1,1,1))
    # for index in range(0, cropped_image_square_shape[-1]):
    #     image_temp = cropped_image_square[:, :, index:index+1].squeeze(-1)
    #     image_temp_nifti = nib.Nifti1Image(image_temp,
    #                                         header = cropped_sqaure_header_2D,
    #                                         affine = cropped_sqaure_affine_2D)
    #     file_name_2D = str(index) + '.nii.gz'
    #     os.makedirs(FolderPath_2D, exist_ok=True)
    #     nib.save(image_temp_nifti, os.path.join(FolderPath_2D, file_name_2D))
    
    if plot_debug:
        # image_numpy = sitk_to_numpy(cropped_image)
        image_numpy = cropped_image
        
        plt.imshow(image_numpy[:, :, 0], cmap='bone')
        plt.title('Cropped End Diastolic Image')
        plt.axis('off')
        plt.show()
        plt.close()
        
        plt.imshow(image_numpy[:, :, 3], cmap='bone')
        plt.title('Cropped End Systolic Image')
        plt.axis('off')
        plt.show()
        plt.close()

    return cropped_image_square

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i","--image", required = True, 
                help = "Cine image saved as numpy array")
# ap.add_argument("--label_dir", required = True, 
                # help = "Train/Val/Test directory")
ap.add_argument("-m","--model", required = True, 
                help = "the model path ends in .pth")
ap.add_argument("-c", "--class", required = False, default = 2, 
                help = "number of class")
ap.add_argument("-s", "--seq", required = False, default = 'cine_lvot_ROI_3D',
                help = "cine_lvot_ROI_3D/cine_2ch_ROI_3D/cine_4ch_ROI_3D")
# It is a 3-chamber input.

args = ap.parse_args()
connection, config, metadata, ds = parse_h5_into_fire_arguments(args.image)
ismrd_images = []
for item in connection:
    if isinstance(item, ismrmrd.Image):
        ismrd_images.append(item)

ismrd_images.remove(ismrd_images[0])
# Finding spatial resolution - see OneNote on 1st May for details and next steps

# for parsing of metadata, see https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/analyzeflow.py
# This does not work, as the matrix size does not match the image shape
imgs = np.array([image.data for image in ismrd_images])
imgs = np.squeeze(imgs)
inishape = imgs.shape
# print(inishape)
imgs = np.swapaxes(imgs, 0, -1)
# print(imgs.shape)

# np.save('CINE3ch.npy', imgs)

# plt.imshow(imgs[10,0,0], cmap='gray')
# plt.show()

cropped_image = detect_roi(imgs,plot_debug=False)

# make sure the input is in Pytorch tensor format
# with dimensions [1, 1, 224, 224, 30]
val_transforms = monai.transforms.Compose(
    [
        monai.transforms.Spacing(pixdim = (1, 1, 1)),
        monai.transforms.Resize(spatial_size = (224, 224, 30)),
        monai.transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
        monai.transforms.EnsureType()
        ]
)

device = torch.device('cpu')
# print('---CUDA STATUS: ', device)
# print('---CUDA VERSION: ', torch.version.cuda)

model = DenseNet201(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = 2,
).to(device)

# layers = medcam.get_layers(model)

sz = cropped_image.shape
norm_image = np.zeros(sz)
for i in range(0,sz[2]):
    ima=cropped_image[:,:,i]
    me=np.mean(ima>0)
    st=np.std(ima>0)
    norm_image[:,:,i] = (ima-me)/st

input_ima = scimage.zoom(norm_image, (224/sz[0], 224/sz[1], 30/sz[2]))
img = np.expand_dims(input_ima, axis=0)
img = np.expand_dims(img, axis=0)
input_t = torch.from_numpy(img).float()
sz2 = img.shape
# print(sz2)

# saved_model = torch.load(os.path.join(args.model), map_location=device)
# Evaluate the model performance
model.load_state_dict(
    torch.load(os.path.join(args.model), map_location=device)
    )

# Classify image    
model.eval()
label_t_all = model(input_t) # take the values too
label_t = label_t_all.argmax(dim=-1)
# weights = label_t_all # ADD ME LATER
label = label_t.detach().numpy().squeeze()
# print(label)

# !!!!! FIX ME !!!!!
# label = 1

# # Make ONNX model
# Make and check .onnx model
# torch_input = torch.randn(sz2)

# Export the model
# torch.onnx.export(model,               # model being run
#                   torch_input,                         # model input (or a tuple for multiple inputs)
#                   'AVAI.onnx',   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                 'output' : {0 : 'batch_size'}})


# onnx_model = onnx.load("AVAI.onnx")
# onnx.checker.check_model(onnx_model)


# Examime the GradCAM headmap
if label == 1:
    cam_model = medcam.inject(model,
                                label = label,
                                layer = 'auto', # which layer is this?
                                data_shape = (224, 224, 30),
                                replace = True)
    cam_img = cam_model(input_t)

    # img_np = input_t.detach().cpu().squeeze().squeeze().numpy()
    cam_img = cam_img.detach().cpu().squeeze().squeeze().numpy()
    # print(cam_img.shape)

    # savemat('outputs.mat', {'cam_img': cam_img, 'img': input_ima})

    
if label == 1:
    show_img = cam_img[:, :, 0]
    text = 'Evidence of Aortic Valve Disease!!!'
else:
    show_img=imgs[:, :, 0]
    text = 'NO evidence of Aortic Valve Disease.'

# create and display the new image with the text
fig, ax = plt.subplots()
ax.imshow(show_img, cmap='gray')
ax.set_title(text)
ax.axis('off')

io_buf = io.BytesIO()
fig.savefig(io_buf, format='raw')
io_buf.seek(0)
npyvec = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
w, h = fig.canvas.get_width_height()
im = npyvec.reshape((int(h), int(w), -1))
io_buf.close()
plt.close()

imgray=np.mean(im[:,:,0:2], axis=2).astype(np.uint8)
# print(np.shape(imgray))
# print(np.max(imgray))
# print(np.min(imgray))

# if show_output == True:
if False:
    plt.imshow(imgray)
    plt.axis('off')
    plt.show()
    plt.close()

imtosave = scimage.zoom(imgray, (inishape[1]/h, inishape[2]/w))
imtosave = np.expand_dims(imtosave, axis=0)
imtosave = np.tile(imtosave, (inishape[0], 1, 1))
# print(np.shape(imtosave))
# dcmimg = sitk.GetImageFromArray(imgray)
# sitk.WriteImage(dcmimg, "Output2.dcm")

# Normalize and convert to int16
maxVal = 2**8 - 1
imtosave = (imtosave.astype(np.float64) - 2048)*maxVal/2048
imtosave = np.around(imtosave).astype(np.int16)

# Create new MRD instance for the processed image
# data has shape [y x sli phs]
# from_array() should be called with 'transpose=False' to avoid warnings, and when called
# with this option, can take input as: [cha z y x], [z y x], or [y x]
tmpImg = ismrmrd.Image.from_array(imtosave, transpose=False)

ismrd_images.remove(ismrd_images[0])
ismrd_images.append(tmpImg)


# Create an MRD file
mrdDset = ismrmrd.Dataset('Output.h5')

# Write MRD Header
mrdDset.write_xml_header(ds.read_xml_header())

# Write all images
for i in range(inishape[0]+1):
    mrdDset.append_image("image_%d" % i, tmpImg)
mrdDset.close()
