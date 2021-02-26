import matplotlib.pyplot as plt


if __name__ == '__main__':

    models = ['I3D', 'TPN', 'TSM', 'SlowFast']
    fig = plt.figure(figsize=(8,5))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    markersize = 80

    # I3D
    I3D_DNN_HMDB = [94.69, 75.07]  # (closed-set ACC, open-set AUC)
    I3D_DNN_MiT = [94.69, 79.14]
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
    # SlowFast
    SlowFast_DNN_HMDB = [96.72, 75.41]
    SlowFast_DNN_MiT = [96.72, 78.49]
    SlowFast_DEAR_HMDB = [96.66, 82.94]
    SlowFast_DEAR_MiT = [96.66, 86.99]

    # Line: DNN for HMDB
    plt.plot([I3D_DNN_HMDB[0], TSM_DNN_HMDB[0], TPN_DNN_HMDB[0], SlowFast_DNN_HMDB[0]],
             [I3D_DNN_HMDB[1], TSM_DNN_HMDB[1], TPN_DNN_HMDB[1], SlowFast_DNN_HMDB[1]], 'r-', linewidth=2, label='HMDB')
    # Line: DEAR for HMDB
    plt.plot([I3D_DEAR_HMDB[0], TSM_DEAR_HMDB[0], TPN_DEAR_HMDB[0], SlowFast_DEAR_HMDB[0]],
             [I3D_DEAR_HMDB[1], TSM_DEAR_HMDB[1], TPN_DEAR_HMDB[1], SlowFast_DEAR_HMDB[1]], 'r-', linewidth=2)
    # Line: DNN for MiT
    plt.plot([I3D_DNN_MiT[0], TSM_DNN_MiT[0], TPN_DNN_MiT[0], SlowFast_DNN_MiT[0]],
             [I3D_DNN_MiT[1], TSM_DNN_MiT[1], TPN_DNN_MiT[1], SlowFast_DNN_MiT[1]], 'b-', linewidth=2, label='MiT')
    # Line: DEAR for MiT
    plt.plot([I3D_DEAR_MiT[0], TSM_DEAR_MiT[0], TPN_DEAR_MiT[0], SlowFast_DEAR_MiT[0]],
             [I3D_DEAR_MiT[1], TSM_DEAR_MiT[1], TPN_DEAR_MiT[1], SlowFast_DEAR_MiT[1]], 'b-', linewidth=2)
    

    # Draw all I3D points
    # HMDB
    plt.scatter(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], marker='^', s=markersize, color='r', label='Dropout BALD')
    plt.text(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], 'I3D', fontsize=fontsize)
    plt.scatter(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], marker='*', s=markersize, color='r', label='DEAR EU')
    plt.text(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], 'I3D', fontsize=fontsize)
    plt.plot([I3D_DNN_HMDB[0], I3D_DEAR_HMDB[0]], [I3D_DNN_HMDB[1], I3D_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(I3D_DNN_HMDB[0]+1, I3D_DNN_HMDB[1], I3D_DEAR_HMDB[0]-I3D_DNN_HMDB[0]-2, I3D_DEAR_HMDB[1]-I3D_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(I3D_DNN_MiT[0], I3D_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(I3D_DNN_MiT[0], I3D_DNN_MiT[1], 'I3D', fontsize=fontsize)
    plt.scatter(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], 'I3D', fontsize=fontsize)
    plt.plot([I3D_DNN_MiT[0], I3D_DEAR_MiT[0]], [I3D_DNN_MiT[1], I3D_DEAR_MiT[1]], 'b--', linewidth=0.5)
    # plt.arrow(I3D_DNN_MiT[0]+1, I3D_DNN_MiT[1], I3D_DEAR_MiT[0]-I3D_DNN_MiT[0]-3, I3D_DEAR_MiT[1]-I3D_DNN_MiT[1]-2,head_width=0.8, fc='grey',ec='grey', head_length=0.8)

    # Draw all TSM points
    # HMDB
    plt.scatter(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], 'TSM', fontsize=fontsize)
    plt.scatter(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], 'TSM', fontsize=fontsize)
    plt.plot([TSM_DNN_HMDB[0], TSM_DEAR_HMDB[0]], [TSM_DNN_HMDB[1], TSM_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(TSM_DNN_HMDB[0]+1, TSM_DNN_HMDB[1], TSM_DEAR_HMDB[0]-TSM_DNN_HMDB[0]-2, TSM_DEAR_HMDB[1]-TSM_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(TSM_DNN_MiT[0], TSM_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(TSM_DNN_MiT[0], TSM_DNN_MiT[1], 'TSM', fontsize=fontsize)
    plt.scatter(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], 'TSM', fontsize=fontsize)
    plt.plot([TSM_DNN_MiT[0], TSM_DEAR_MiT[0]], [TSM_DNN_MiT[1], TSM_DEAR_MiT[1]], 'b--', linewidth=0.5)

    # Draw all TPN points
    # HMDB
    plt.scatter(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], 'TPN', fontsize=fontsize)
    plt.scatter(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], 'TPN', fontsize=fontsize)
    plt.plot([TPN_DNN_HMDB[0], TPN_DEAR_HMDB[0]], [TPN_DNN_HMDB[1], TPN_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(TPN_DNN_HMDB[0]+1, TPN_DNN_HMDB[1], TPN_DEAR_HMDB[0]-TPN_DNN_HMDB[0]-2, TPN_DEAR_HMDB[1]-TPN_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    plt.scatter(TPN_DNN_MiT[0], TPN_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(TPN_DNN_MiT[0], TPN_DNN_MiT[1], 'TPN', fontsize=fontsize)
    plt.scatter(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], 'TPN', fontsize=fontsize)
    plt.plot([TPN_DNN_MiT[0], TPN_DEAR_MiT[0]], [TPN_DNN_MiT[1], TPN_DEAR_MiT[1]], 'b--', linewidth=0.5)

    # Draw all SlowFast points
    # HMDB
    plt.scatter(SlowFast_DNN_HMDB[0], SlowFast_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(SlowFast_DNN_HMDB[0], SlowFast_DNN_HMDB[1], 'SlowFast', fontsize=fontsize)
    plt.scatter(SlowFast_DEAR_HMDB[0], SlowFast_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(SlowFast_DEAR_HMDB[0], SlowFast_DEAR_HMDB[1], 'SlowFast', fontsize=fontsize)
    plt.plot([SlowFast_DNN_HMDB[0], SlowFast_DEAR_HMDB[0]], [SlowFast_DNN_HMDB[1], SlowFast_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # MiT
    plt.scatter(SlowFast_DNN_MiT[0], SlowFast_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(SlowFast_DNN_MiT[0], SlowFast_DNN_MiT[1], 'SlowFast', fontsize=fontsize)
    plt.scatter(SlowFast_DEAR_MiT[0], SlowFast_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(SlowFast_DEAR_MiT[0], SlowFast_DEAR_MiT[1], 'SlowFast', fontsize=fontsize)
    plt.plot([SlowFast_DNN_MiT[0], SlowFast_DEAR_MiT[0]], [SlowFast_DNN_MiT[1], SlowFast_DEAR_MiT[1]], 'b--', linewidth=0.5)
    
    plt.xlim(94, 97.5)
    plt.ylim(65, 90)
    plt.xlabel('Closed-Set Accuracy (%)', fontsize=fontsize)
    plt.ylabel('Open-Set AUC Score (%)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='lower left', fontsize=fontsize)
    plt.grid('on', linestyle='--')
    plt.savefig('../temp/compare_gain.png')
    plt.savefig('../temp/compare_gain.pdf')