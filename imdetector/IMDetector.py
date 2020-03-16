import re
import os
import glob
import shutil
import datetime
import cv2 as cv
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
from tkinter import messagebox as mess

# from dismantler import Dismantler
from imgminer import imgminer
from dismantler import Dismantler
from image import SuspiciousImage
from photopick import PhotoPick
from clipping import Clipping
from copymove import CopyMove, Duplication
from noise import Noise
from plot import plot_pdfpages


class AppForm(tk.Frame):
    def __init__(self, master=None):
        # super().__init__(master, height=500, width=660)
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.master = master
        # self.menubar_create()

    def create_widgets(self):
        # Wrapper frame
        wrpFrm = tk.Frame(self.master, bg='white')
        wrpFrm.pack(padx=5, pady=5, fill='x', expand=1)

        # Image list
        imgFrm = tk.LabelFrame(
            wrpFrm, bg='white', text='Check List', font=('MS Sans Serif', 16, 'bold'))
        imgFrm.pack(padx=5, pady=5, ipadx=5, ipady=2, fill='both',
                    anchor=tk.W, side=tk.TOP, expand=1)
        self.imgList = ttk.Treeview(imgFrm)
        self.imgList.configure(
            column=(1, 2), show='headings', height=6)
        self.imgList.column(1, width=30)
        self.imgList.column(2, width=570)
        self.imgList.heading(1, text='No')
        self.imgList.heading(2, text='path/name')
        self.imgList.pack(padx=5, pady=5)
        self.f_path_list = []
        # Buttons
        btnFrm = tk.Frame(imgFrm, bg='white')
        btnFrm.pack(ipadx=5, ipady=5)
        addBtn = ttk.Button(btnFrm, style='blackt.TButton',
                            text='Add image', command=self.img_add)
        addBtn.pack(side='left', padx=5)
        resetBtn = ttk.Button(btnFrm, style='blackt.TButton',
                              text='Reset', command=self.img_reset)
        resetBtn.pack(side='left', padx=5)

        # Save path
        acvFrm = tk.LabelFrame(
            wrpFrm, bg='white', text='Destination Directory', font=('MS Sans Serif', 16, 'bold'))
        acvFrm.pack(padx=5, pady=5, ipadx=10, ipady=2, fill='both')
        # acvFrm = tk.Frame(wrpFrm, bg='white')
        # acvFrm.pack(ipadx=5, ipady=10)
        acvTtlLbl = tk.Label(acvFrm, bg='white', text='Save in: ')
        acvTtlLbl.pack(padx=5, pady=5, side='left')
        self.acvEnt = tk.Entry(acvFrm, width=40)
        self.acvEnt.insert(0, 'Unselected')
        self.acvEnt.configure(state='disabled')
        self.acvEnt.pack(side='left', fill=tk.X)
        acvBtn = ttk.Button(acvFrm, style='blackt.TButton',
                            text='Browse', command=self.acv_open)
        acvBtn.pack(padx=5, pady=5, side='left', fill=tk.X)

        # Options
        runFrm = tk.Frame(wrpFrm, bg='white')
        runFrm.pack(fill='both')
        optionFrm = tk.LabelFrame(
            runFrm, bg='white', text='Option', font=('MS Sans Serif', 12))
        optionFrm.pack(padx=5, pady=5, ipadx=10, ipady=2,
                       fill='both', side=tk.LEFT)
        booleanGridFrm = tk.Frame(optionFrm, bg='white')
        booleanGridFrm.pack(padx=5, pady=5, ipadx=10, ipady=2, fill='both')
        # crop?
        self.cropChkVar = tk.BooleanVar()
        self.cropChkVar.set(True)
        cropChk = ttk.Checkbutton(booleanGridFrm)
        cropChk.configure(
            text='Cut out margins from figures',
            variable=self.cropChkVar,
        )
        cropChk.grid(row=0, column=0, sticky='nw', padx=5, pady=5)
        # classify?
        self.classifyChkVar = tk.BooleanVar()
        self.classifyChkVar.set(True)
        classifyChk = ttk.Checkbutton(booleanGridFrm)
        classifyChk.configure(
            text='Select life sciences images from figures',
            variable=self.classifyChkVar,
        )
        classifyChk.grid(row=1, column=0, sticky='nw', padx=5, pady=5)
        # histogram equalization?
        self.histChkVar = tk.BooleanVar()
        self.histChkVar.set(True)
        histChk = ttk.Checkbutton(booleanGridFrm)
        histChk.configure(
            text='Apply histogram equalization to images',
            variable=self.histChkVar,
        )
        histChk.grid(row=2, column=0, sticky='nw', padx=5, pady=5)

        optionGridFrm = tk.Frame(optionFrm, bg='white')
        optionGridFrm.pack(padx=5, pady=5, ipadx=10, ipady=2, fill='both')
        # keypoint algorithm
        # cmLbl = tk.Label(optionGridFrm, bg='white',
        #                  text='Copy-move', font=('MS Sans Serif', 14))
        # cmLbl.grid(row=0, column=0, sticky='w', padx=5, pady=1)
        kpLbl = tk.Label(optionGridFrm, bg='white',
                         text='Keypoint algorithm:', font=('MS Sans Serif', 12))
        kpLbl.grid(row=0, column=1, sticky='e', padx=5, pady=1)
        self.kpSelect = ttk.Combobox(optionGridFrm)
        self.kpSelect.configure(
            state='readonly',
            width=8,
            value=('ORB', 'AKAZE'),
            font=('MS Sans Serif', 12))
        self.kpSelect.current(0)
        self.kpSelect.grid(row=0, column=2, sticky='w', padx=5, pady=1)
        # nfeatures
        kpnLbl = tk.Label(optionGridFrm, bg='white',
                          text='Maximum number of keypoints:', font=('MS Sans Serif', 12))
        kpnLbl.grid(row=1, column=1, sticky='e', padx=5, pady=1)
        self.kpnEnt = tk.Scale(optionGridFrm, orient='h',
                               from_=0, to=10000, length=200, font=('MS Sans Serif', 10))
        self.kpnEnt.set(2000)
        self.kpnEnt.grid(row=1, column=2, sticky='w', padx=5, pady=5)

        # Run
        runBtn = ttk.Button(runFrm, style='run.TButton',
                            text='Run', command=self.img_run)
        runBtn.pack(
            anchor='e',
            pady=(0, 5), padx=(5, 5),
            ipadx=10, ipady=8, side=tk.BOTTOM, fill=tk.X)

    def img_add(self):
        f_conf = [('Text Files', ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                                  '.tif', '.TIF', '.tiff', '.TIFF', '.bmp', '.BMP', '.pdf', '.PDF'))]
        paths = tkfd.askopenfiles(filetypes=f_conf)
        for f in paths:
            self.imgList.insert('', 'end', values=(
                len(self.f_path_list), f.name))
            self.f_path_list.append(f.name)

    def img_reset(self):
        for i in self.imgList.get_children():
            self.imgList.delete(i)
        self.f_path_list = []

    def acv_open(self):
        self.acv_dir = tkfd.askdirectory()
        self.acvEnt.configure(state='normal')
        self.acvEnt.delete(0, 'end')
        if self.acv_dir != '':
            self.acvEnt.insert(0, self.acv_dir)
        else:
            self.acvEnt.insert(0, 'unselected')
        self.acvEnt.configure(state='disabled')

    def img_run(self):
        # Validation
        if len(self.f_path_list) == 0:
            mess.showwarning('Warning', 'Select images to check.')
            return None
        if not hasattr(self, 'acv_dir'):
            mess.showwarning(
                'Warning', 'Select the folder where you would like to save results.')
            return None
        try:
            pdf_list = [
                f for f in self.f_path_list if f.endswith(('.pdf', '.PDF'))]
            self.f_path_list = list(set(self.f_path_list) - set(pdf_list))

            date = datetime.datetime.now()
            self.tmp_dir = os.path.join(self.acv_dir, 'imdetector_tmp')
            os.makedirs(self.tmp_dir, exist_ok=True)

            # Extract images
            for f in pdf_list:
                self.extraction(f)
            if len(self.f_path_list) == 0:
                mess.showwarning(
                    'Warning', 'No images were extracted. Please upload image files.')
                return None

            # Split and crop images
            if self.cropChkVar.get():
                Dismantler().dismantle(self.f_path_list, self.tmp_dir)
                self.f_path_list = glob.glob(os.path.join(
                    self.tmp_dir, 'subimg_cut', '*.png'))

            suspicious_images = [SuspiciousImage(
                path, hist_eq=self.histChkVar.get(), algorithm=self.kpSelect.get().lower(), nfeatures=int(self.kpnEnt.get()), gap=32) for path in self.f_path_list]

            # Classify images
            if self.classifyChkVar.get():
                detector = PhotoPick()
                pred = detector.detect(suspicious_images)
                suspicious_images = [img for img, p in zip(
                    suspicious_images, pred) if p]

            # Detectors
            detector_cl = Clipping()
            detector_cm = CopyMove(min_kp=20, min_match=20, min_key_ratio=0.75)
            detector_du = Duplication(
                min_kp=20, min_match=20, min_key_ratio=0.75)
            # detector_no = Noise(model_name='model/noise_oneclass_42.sav')
            self.result_dir = os.path.join(self.acv_dir, 'imdetector_result')
            os.makedirs(self.result_dir, exist_ok=True)
            len_sus = len(suspicious_images)

            # Report
            report = pd.DataFrame(
                0,
                index=list(range(len_sus)),
                columns=[
                    'Name',
                    'Clipping',
                    'area_ratio',
                    'CopyMove',
                    'mask_ratio',
                    'Duplication', ])
            report['Name'] = [img.name for img in suspicious_images]
            report['Duplication'] = ['X']*report.shape[0]

            for i in range(len_sus):
                img = suspicious_images[i]
                # imgname = img.name
                imgname = i

                # Clipping check #
                pred = detector_cl.detect(img)
                img.clipping = pred
                if pred is 1:
                    ratio = detector_cl.ratio_
                    img.area_ratio = ratio
                    img.cl_img = detector_cl.image_
                report.loc[imgname, 'Clipping'] = pred
                report.loc[imgname, 'area_ratio'] = img.area_ratio

                # Copy-move check #
                pred = detector_cm.detect(img)
                img.copymove = pred
                if pred is 1:
                    ratio = detector_cm.mask_ratio_
                    img.mask_ratio = ratio
                    img.cm_img = detector_cm.image_
                report.loc[imgname, 'CopyMove'] = pred
                report.loc[imgname, 'mask_ratio'] = img.mask_ratio

            # Noise check #
            # report['Noise'] = detector_no.detect(suspicious_images)
            # report['dist'] = detector_no.dist_
            # for i, img in enumerate(suspicious_images):
            #     img.noise = report.Noise[i]

            # Duplication check #
            flip_images = [
                SuspiciousImage().make_flip(img.mat, img.name + '-flip')
                for img in suspicious_images]
            result_imgnames = []
            result_imgarrs = []
            result_ratios = []
            for i in range(len_sus):
                img = suspicious_images[i]
                # imgname = img.name
                imgname = i
                for j in range(i + 1, len_sus):
                    pred = detector_du.detect(
                        img, suspicious_images[j])
                    if pred is 1:
                        report.loc[imgname, 'Duplication'] = 'O'
                        result_imgnames.append(
                            [imgname, j])
                        result_imgarrs.append(detector_du.image_)
                        result_ratios.append(detector_du.mask_ratio_)

                    # flipped images
                    pred = detector_du.detect(
                        img, flip_images[j])
                    if pred is 1:
                        report.loc[imgname, 'Duplication'] = 'O'
                        result_imgnames.append(
                            [imgname, str(j)+'_flip'])
                        result_imgarrs.append(detector_du.image_)
                        result_ratios.append(detector_du.mask_ratio_)

            # Output report #
            plot_pdfpages(suspicious_images, result_imgnames,
                          result_imgarrs, result_ratios, date, self.result_dir)

            datestr = date.strftime('%Y-%m-%d-%H%M%S')
            report.to_csv(os.path.join(self.result_dir,
                                       '{}-report.csv'.format(datestr)))
            self.img_dir = os.path.join(
                self.result_dir, '{}-image'.format(datestr))
            os.makedirs(self.img_dir, exist_ok=True)
            [cv.imwrite(os.path.join(self.img_dir, str(i)+'.png'),
                        suspicious_images[i].mat) for i in range(len_sus)]

            shutil.rmtree(self.tmp_dir)
            mess.showinfo(
                'IMDetector', 'The operation successfully completed!\n\nSaved in: {}'.format(self.result_dir))
        except:
            mess.showerror(
                'IMDetector', 'The operation failed.')
        self.img_reset()
        return None

    def extraction(self, pdf_file):
        extract_img_dir = os.path.join(self.tmp_dir, 'img')
        os.makedirs(extract_img_dir, exist_ok=True)
        imgminer(pdf_file, OUT_DIR=extract_img_dir)
        self.f_path_list = self.f_path_list + \
            glob.glob(os.path.join(extract_img_dir, '*.png'))


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Image Manipulation Detector')
    root.geometry('650x550')
    root.option_add('*font', ('MS Sans Serif', 16))
    root.option_add('*foreground', 'black')
    ttk.Style().configure('blackt.TButton', foreground='black',
                          background='white', font=('MS Sans Serif', 16))
    ttk.Style().configure('run.TButton', foreground='black',
                          background='white', font=('MS Sans Serif', 20, 'bold'))
    app = AppForm(master=root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
