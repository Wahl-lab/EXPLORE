# ---------------------------------------------------------------------
#          QUARTER ANALYSIS / 07.02.21, Victor Ibanez
# ---------------------------------------------------------------------

# import library
import cv2
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import seaborn as sns
import math

def quadrant_analysis(project_path, project_name, videos, time, background, ref_point): # plot,

    def analysis(video, name):

        # extract light intensity from videos
        print('Processing...')
        vidcap = cv2.VideoCapture(video)
        success,image = vidcap.read()
        fps =  vidcap.get(cv2.CAP_PROP_FPS)
        max_time = (int(time)*60*int(fps))+15
        count = 1

        NW_l = []
        SW_l = []
        NE_l = []
        SE_l = []

        while success:

            h,w,d = image.shape
            res1 = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] # crop video
            res = cv2.resize(res1, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
      
            y,x,channels = res.shape
            #res = res/255.
            y_h = int(round(y/2))
            x_h = int(round(x/2))

            # take mean from each quarter
            if count >= 15 and count < max_time:
                img1 = res[:y_h,:x_h]
                mean1 = round(np.mean(img1),4)
                NW_l.append(mean1)
                img2 = res[y_h:y,:x_h]
                mean2 = round(np.mean(img2),4)
                SW_l.append(mean2)
                img3 = res[:y_h,x_h:x]
                mean3 = round(np.mean(img3),4)
                NE_l.append(mean3)
                img4 = res[y_h:y,x_h:x]
                mean4 = round(np.mean(img4),4)
                SE_l.append(mean4)

            elif count > max_time:
                break    

            success,image = vidcap.read()
            count += 1

        #-------------------------------------------------------------------------
        # get max of each quarter
        def min_max(vector):
            vec_min = min(vector[1000:-1000])
            vec_max = max(vector[1000:-1000])
            return vec_min, vec_max

        min_NW, max_NW = min_max(NW_l)
        min_NE, max_NE = min_max(NE_l)
        min_SW, max_SW = min_max(SW_l)
        min_SE, max_SE = min_max(SE_l)

        def norm(vector, vec_min, vec_max):
            vec_norm = (vector-vec_min)/(vec_max-vec_min)
            return vec_norm

        # normalize quarters
        NW_norm = norm(NW_l, min_NW, max_NW)
        NE_norm = norm(NE_l, min_NE, max_NE)
        SW_norm = norm(SW_l, min_SW, max_SW)
        SE_norm = norm(SE_l, min_SE, max_SE)

        # set threshold
        theta = 0.5

        # threshold function
        def threshold(vec_old, theta):
            vec_new = []
            if background == 1:
                inv = -1
            else:
                inv = 1
            for i in vec_old:
                x = (1/2*np.sign(i-theta)-1/2)*inv
                vec_new.append(x)
            return vec_new

        NW_thr = threshold(NW_norm, theta)
        NE_thr = threshold(NE_norm, theta)
        SW_thr = threshold(SW_norm, theta)
        SE_thr = threshold(SE_norm, theta)

        t = int(fps)*60
        delta = int((len(NW_thr))/int(time))

        QET_NW_sum = []
        QET_NE_sum = []
        QET_SW_sum = []
        QET_SE_sum = []
        QEF_NW_sum = []
        QEF_NE_sum = []
        QEF_SW_sum = []
        QEF_SE_sum = []

        for i in range(0,len(NW_thr), t):

            # exctract QET
            QET_NW = sum(1 for x in range(i,i+delta) if NW_thr[x] == 1)
            QET_NE = sum(1 for x in range(i,i+delta) if NE_thr[x] == 1)
            QET_SW = sum(1 for x in range(i,i+delta) if SW_thr[x] == 1)
            QET_SE = sum(1 for x in range(i,i+delta) if SE_thr[x] == 1)
            QET_NW_sum.append(QET_NW/int(fps))
            QET_NE_sum.append(QET_NE/int(fps))
            QET_SW_sum.append(QET_SW/int(fps))
            QET_SE_sum.append(QET_SE/int(fps))

            # exctract QEF
            QEF_NW = sum(1 for x in range(i,(i+delta)-1) if NW_thr[x] == 1 and NW_thr[x] != NW_thr[x+1])
            QEF_NE = sum(1 for x in range(i,(i+delta)-1) if NE_thr[x] == 1 and NE_thr[x] != NE_thr[x+1])
            QEF_SW = sum(1 for x in range(i,(i+delta)-1) if SW_thr[x] == 1 and SW_thr[x] != SW_thr[x+1])
            QEF_SE = sum(1 for x in range(i,(i+delta)-1) if SE_thr[x] == 1 and SE_thr[x] != SE_thr[x+1])
            QEF_NW_sum.append(QEF_NW)
            QEF_NE_sum.append(QEF_NE)
            QEF_SW_sum.append(QEF_SW)
            QEF_SE_sum.append(QEF_SE)

            # append to data frame
            d = {'experiment':Exp_name,'animal':name,'minute':(i/int(fps)/60)+1,'QET_NW':QET_NW/int(fps),
            'QET_NE':QET_NE/int(fps), 'QET_SW':QET_SW/int(fps), 'QET_SE':QET_SE/int(fps), 'QEF_NW':QEF_NW,
            'QEF_NE':QEF_NE, 'QEF_SW':QEF_SW, 'QEF_SE':QEF_SE}
            data.append(d)

        #if plot == 1:
            # plot thresholded data
        fig, axs = plt.subplots(4, sharex=True, sharey=True,figsize=(30,5))
        axs[0].tick_params(axis='x', which='major', labelsize=16)
        axs[0].tick_params(axis='y', which='major', labelsize=16)
        axs[0].plot(NW_thr)
        axs[0].set_title('quadrant analysis', fontsize=20)
        axs[0].legend(['NW'],loc='center left', fontsize=14)
        axs[1].tick_params(axis='x', which='major', labelsize=16)
        axs[1].tick_params(axis='y', which='major', labelsize=16)
        axs[1].plot(NE_thr, color='g')
        axs[1].legend(['NE'],loc='center left', fontsize=14)
        axs[2].tick_params(axis='x', which='major', labelsize=16)
        axs[2].tick_params(axis='y', which='major', labelsize=16)
        axs[2].plot(SW_thr, color='r')
        axs[2].legend(['SW'],loc='center left', fontsize=14)
        axs[3].tick_params(axis='x', which='major', labelsize=16)
        axs[3].tick_params(axis='y', which='major', labelsize=16)
        axs[3].plot(SE_thr, color='c')
        axs[3].legend(['SE'],loc='center left', fontsize=14)
        plt.savefig(plt_path + '/QET_' + name + '.png')
        plt.close()

        d2 = {'experiment':Exp_name,'animal':name,'QET_left':np.sum(QET_NW_sum),'QET_right':np.sum(QET_NE_sum),
              'QEF_left':np.sum(QEF_NW_sum),'QEF_right':np.sum(QEF_NE_sum)}
        data_HM.append(d2)
        d3 = {'experiment':Exp_name,'animal':name,'QET_left':np.sum(QET_SW_sum),'QET_right':np.sum(QET_SE_sum),
              'QEF_left':np.sum(QEF_SW_sum),'QEF_right':np.sum(QEF_SE_sum)}
        data_HM.append(d3)
    #-------------------------------------------------------------------------
    # extract QET / QEF

    Exp_name = os.path.basename(os.path.dirname(videos[0]))

    #if plot == 1:
    plt_path = os.path.join(project_path,project_name,'plots')
    if not os.path.isdir(plt_path):
        os.makedirs(plt_path)

    data = []
    data_HM = []

    cnt = 1
    for v in videos:
        split = v.split('/')
        vid_name = split[-1]
        analysis(v, vid_name)
        print('processed video:', vid_name)
        cnt += 1


    # write a csv with all values
    def write(dataframe, name):
        df = pd.DataFrame.from_dict(dataframe)
        filename = os.path.join(project_path,project_name,name+ Exp_name + '.csv')

        if os.path.isfile(filename):
            cnt = 1
            while True:
                new_filename = os.path.join(project_path,project_name,name+ Exp_name + '_' + str(cnt) + '.csv')
                cnt += 1
                if os.path.isfile(new_filename):
                    continue
                else:
                    filename = new_filename
                    break
        df.to_csv(filename, index=False)

    df1 = pd.DataFrame.from_dict(data)
    df2 = pd.DataFrame.from_dict(data_HM)
    name1 = 'Quadrant_analysis_'
    #name2 = 'HM_Quadrant_analysis_'
    write(df1,name1)
    #write(df2,name2)

    if len(videos) > 1: # plot == 1 and
        def heatmap(df, type, type2, name_HM):
            cols = math.ceil(np.sqrt(len(df)/2))
            rows = math.floor(np.sqrt(len(df)/2))
            fig, axs = plt.subplots(rows, cols, sharex=False, sharey=False)
            axs = axs.ravel()
            cmap = sns.cm.rocket
            norm = mpl.colors.Normalize(vmin=np.min(df), vmax=np.max(df))
            n = 2
            for i in range(rows*cols):
                if i >= int(len(df)/2):
                    fig.delaxes(axs.flatten()[i])
                im = axs[i].imshow(df[(i*n):(i*n)+2], cmap=cmap , norm=norm, interpolation='nearest')
                axs[i].set_ylabel(str(i+1), c='k',rotation=0,va='center',labelpad=5)
                axs[i].set_xticks([])
                axs[i].set_yticks([])

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            plt.colorbar(im, cbar_ax)
            fig.suptitle(name_HM+'_'+type, fontsize=12)
            plt.title(type2)
            plt.savefig(os.path.join(plt_path, name_HM + '_' + type + '.png'), bbox_inches='tight',format='png')
            fig.clf()

        HM1 = df2.iloc[:,2:4]
        HM2 = df2.iloc[:,4:6]
        heatmap(np.asarray(HM1),'QET','(s)',project_name)
        heatmap(np.asarray(HM2),'QEF','(f)',project_name)



