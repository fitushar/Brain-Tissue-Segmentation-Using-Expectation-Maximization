# Brain-Tissue-Segmentation-Using-Expectation-Maximization
Medical Image Segmentation and Applications (MISA) LAB task.

Functions Used in two codes::

1. show_slice(img, slice_no):
    
    Inputs: Name of the Image Array, img=name.get_fdata()
            Slice number you want to knoe,Slice no = 24
    output: Plot Image.
2. gmm(x, mean, cov):
    
    Inputs:
        x (numpy.ndarray): nxd dimentional array. where n= number of samples
                                                        d= dimention
        mean (numpy.ndarray): d-dimentional mean value.
        cov (numpy.ndarray): dxd dimentional covariance matrix.
    
    output:
        (numpy.ndarray): Gaussian mixture for every point in feature space.
        
3.dice_similarity(Seg_img, GT_img,state):
     Inputs:
        Seg_img (numpy.ndarray): Segmented Image.
        GT_img (numpy.ndarray): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    output:
        Dice Similarity Coefficient: dice_CSF, dice_GM, dice_WM.
        
4. Dice_and_Visualization_of_one_slice(Seg_img, GT_img,state,number_of_slice):
    """Example Use: Dice_and_Visualization_of_one_slice(Seg,Label_img,"arr",30)"""
         
     Inputs:
        Seg_img (numpy.ndarray): Segmented Image.
        GT_img (numpy.ndarray): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    output:
        Dice Similarity Coefficient: dice_CSF, dice_GM, dice_WM.
        Ploting image
