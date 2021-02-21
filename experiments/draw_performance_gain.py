import matplotlib.pyplot as plt


if __name__ == '__main__':

    models = ['I3D', 'TPN', 'TSM']
    fig = plt.figure(figsize=(8,5))
    plt.rcParams["font.family"] = "Times New Roman"

    # I3D
    I3D_DNN_HMDB = [89.09, 73.59]  # (closed-set ACC, open-set AUC)
    I3D_DNN_MiT = [89.09, 76.69]
    I3D_DEAR_HMDB = [94.34, 77.08]
    I3D_DEAR_MiT = [94.34, 81.54]
    # TSM
    TSM_DNN_HMDB = [95.11, 73.85]
    TSM_DNN_MiT = [95.11, 78.35]
    TSM_DEAR_HMDB = [94.45, 78.65]
    TSM_DEAR_MiT = [94.45, 83.92]
    # TPN
    TPN_DNN_HMDB = [95.41, 74.13]
    TPN_DNN_MiT = [95.41, 77.76]
    TPN_DEAR_HMDB = [96.42, 79.23]
    TPN_DEAR_MiT = [96.42, 81.80]

    # Line: DNN for HMDB
    plt.plot([I3D_DNN_HMDB[0], TSM_DNN_HMDB[0], TPN_DNN_HMDB[0]],
             [I3D_DNN_HMDB[1], TSM_DNN_HMDB[1], TPN_DNN_HMDB[1]], 'r-', linewidth=1, label='HMDB')
    # Line: DEAR for HMDB
    plt.plot([I3D_DEAR_HMDB[0], TSM_DEAR_HMDB[0], TPN_DEAR_HMDB[0]],
             [I3D_DEAR_HMDB[1], TSM_DEAR_HMDB[1], TPN_DEAR_HMDB[1]], 'r-', linewidth=1)
    # Line: DNN for MiT
    plt.plot([I3D_DNN_MiT[0], TSM_DNN_MiT[0], TPN_DNN_MiT[0]],
             [I3D_DNN_MiT[1], TSM_DNN_MiT[1], TPN_DNN_MiT[1]], 'b-', linewidth=1, label='MiT')
    # Line: DEAR for MiT
    plt.plot([I3D_DEAR_MiT[0], TSM_DEAR_MiT[0], TPN_DEAR_MiT[0]],
             [I3D_DEAR_MiT[1], TSM_DEAR_MiT[1], TPN_DEAR_MiT[1]], 'b-', linewidth=1)
    

    # Draw all I3D points
    # HMDB
    plt.scatter(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], marker='^', color='r', label='Dropout BALD')
    plt.text(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], 'I3D')
    plt.scatter(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], marker='*', color='r', label='DEAR EU')
    plt.text(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], 'I3D')
    # plt.arrow(I3D_DNN_HMDB[0]+1, I3D_DNN_HMDB[1], I3D_DEAR_HMDB[0]-I3D_DNN_HMDB[0]-2, I3D_DEAR_HMDB[1]-I3D_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(I3D_DNN_MiT[0], I3D_DNN_MiT[1], marker='^', color='b')
    plt.text(I3D_DNN_MiT[0], I3D_DNN_MiT[1], 'I3D')
    plt.scatter(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], marker='*', color='b')
    plt.text(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], 'I3D')
    # plt.arrow(I3D_DNN_MiT[0]+1, I3D_DNN_MiT[1], I3D_DEAR_MiT[0]-I3D_DNN_MiT[0]-3, I3D_DEAR_MiT[1]-I3D_DNN_MiT[1]-2,head_width=0.8, fc='grey',ec='grey', head_length=0.8)

    # Draw all TSM points
    # HMDB
    plt.scatter(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], marker='^', color='r')
    plt.text(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], 'TSM')
    plt.scatter(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], marker='*', color='r')
    plt.text(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], 'TSM')
    # plt.arrow(TSM_DNN_HMDB[0]+1, TSM_DNN_HMDB[1], TSM_DEAR_HMDB[0]-TSM_DNN_HMDB[0]-2, TSM_DEAR_HMDB[1]-TSM_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(TSM_DNN_MiT[0], TSM_DNN_MiT[1], marker='^', color='b')
    plt.text(TSM_DNN_MiT[0], TSM_DNN_MiT[1], 'TSM')
    plt.scatter(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], marker='*', color='b')
    plt.text(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], 'TSM')

    # Draw all TPN points
    # HMDB
    plt.scatter(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], marker='^', color='r')
    plt.text(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], 'TPN')
    plt.scatter(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], marker='*', color='r')
    plt.text(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], 'TPN')
    # plt.arrow(TPN_DNN_HMDB[0]+1, TPN_DNN_HMDB[1], TPN_DEAR_HMDB[0]-TPN_DNN_HMDB[0]-2, TPN_DEAR_HMDB[1]-TPN_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    plt.scatter(TPN_DNN_MiT[0], TPN_DNN_MiT[1], marker='^', color='b')
    plt.text(TPN_DNN_MiT[0], TPN_DNN_MiT[1], 'TPN')
    plt.scatter(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], marker='*', color='b')
    plt.text(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], 'TPN')
    

    
    plt.xlim(84, 100)
    plt.ylim(65, 90)
    plt.xlabel('Closed-Set Accuracy (%)')
    plt.ylabel('Open-Set AUC Score (%)')
    plt.legend(loc='lower left')
    plt.grid('on', linestyle='--')
    plt.savefig('../temp/compare_gain.png')