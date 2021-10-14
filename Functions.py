def GetNonPerfusionPatch():
    #ClassMask0
    NonPerfusionPatch_Locations = np.argwhere(CLASS_MASK_0==1)
    return NonPerfusionPatch_Locations

def GetPerfusionPatch():
    #ClassMask3
    PerfusionPatch_Locations = np.argwhere(CLASS_MASK_3==4)
    return PerfusionPatch_Locations

def CoOccurenceMatrix():
    # select some patches from NonPerfusion Area of the image
    NonPerfusionPatch_Locations = GetNonPerfusionPatch()
    NumberOfNPLocations = len(NonPerfusionPatch_Locations)
    MidPointOfNPLocations = int(float(len(NonPerfusionPatch_Locations)/2))
    NonPerfusion_Patches = []
    for i in range(0, len(NonPerfusionPatch_Locations)):
        for loc in NonPerfusionPatch_Locations:
            if i == 0:
                NonPerfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
                i += 1
            elif i == MidPointOfNPLocations:
                NonPerfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
                i += 1
            elif i == NumberOfNPLocations:
                NonPerfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
                i += 1
            else:
                i += 1

    # select some patches from Perfusion Area of the image
    PerfusionPatch_Locations = GetPerfusionPatch()
    NumberOfPPLocations = len(PerfusionPatch_Locations)
    MidPointOfPPLocations = int(float(len(PerfusionPatch_Locations)/2))
    Perfusion_Patches = []
    for i in range(0, len(PerfusionPatch_Locations)):
        for loc in PerfusionPatch_Locations:
            if i == 0:
                Perfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
                i += 1
            elif i == MidPointOfPPLocations:
                Perfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
                i += 1
            elif i == NumberOfPPLocations:
                Perfusion_Patches.append(IMG[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])
            #     i += 1
            else:
                i += 1
    
    # compute some GLCM properties each patch
    Dissimilarity = []
    Correlation = []
    for patch in (NonPerfusion_Patches + Perfusion_Patches):
        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        Dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        Correlation.append(greycoprops(glcm, 'correlation')[0, 0])

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(IMG, cmap=plt.cm.gray,
            vmin=0, vmax=255)
    for (y, x) in NonPerfusionPatch_Locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    for (y, x) in PerfusionPatch_Locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(Dissimilarity[:len(NonPerfusion_Patches)], Correlation[:len(NonPerfusion_Patches)], 'go',
            label='NonPerfusion')
    ax.plot(Dissimilarity[len(NonPerfusion_Patches):], Correlation[len(NonPerfusion_Patches):], 'bo',
            label='Perfusion')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(NonPerfusion_Patches):
        ax = fig.add_subplot(3, len(NonPerfusion_Patches), len(NonPerfusion_Patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                vmin=0, vmax=255)
        ax.set_xlabel('NonPerfusion %d' % (i + 1))

    for i, patch in enumerate(Perfusion_Patches):
        ax = fig.add_subplot(3, len(Perfusion_Patches), len(Perfusion_Patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                vmin=0, vmax=255)
        ax.set_xlabel('Perfusion %d' % (i + 1))


    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()