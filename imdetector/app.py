import re
import os
import datetime
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
from tkinter import messagebox as mess

from image import SuspiciousImage
from clipping import Clipping
from copymove import CopyMove
from plot import plot_pdfpages


class AppForm(tk.Frame):
    def __init__(self, master=None):
        # super().__init__(master, height=500, width=660)
        super().__init__(master)
        self.pack()
        self.create_widgets()
        # self.menubar_create()

    def create_widgets(self):
        # Wrapper frame
        wrpFrm = tk.Frame(root, bg="white")
        wrpFrm.pack(padx=5, pady=5, fill="both", expand=1)

        # Image list
        imgFrm = tk.LabelFrame(
            wrpFrm, bg="white", text="Check List", font=('MS Sans Serif', 16, 'bold'))
        imgFrm.pack(padx=5, pady=5, ipadx=5, ipady=2, fill="both",
                    anchor=tk.W, side=tk.TOP, expand=1)
        self.imgList = ttk.Treeview(imgFrm)
        self.imgList.configure(
            column=(1, 2), show="headings", height=6)
        self.imgList.column(1, width=30)
        self.imgList.column(2, width=570)
        self.imgList.heading(1, text="No")
        self.imgList.heading(2, text="path/name")
        self.imgList.pack(padx=5, pady=5)
        self.f_path_list = []
        # Buttons
        btnFrm = tk.Frame(imgFrm, bg="white")
        btnFrm.pack(ipadx=5, ipady=5)
        addBtn = ttk.Button(btnFrm, style="blackt.TButton",
                            text="Add images", command=self.img_add)
        addBtn.pack(side="left", padx=5)
        resetBtn = ttk.Button(btnFrm, style="blackt.TButton",
                              text="Reset", command=self.img_reset)
        resetBtn.pack(side="left", padx=5)

        # Save path
        acvFrm = tk.LabelFrame(
            wrpFrm, bg="white", text="Destination Directory", font=('MS Sans Serif', 16, 'bold'))
        acvFrm.pack(padx=5, pady=5, ipadx=10, ipady=2, fill="both")
        # acvFrm = tk.Frame(wrpFrm, bg="white")
        # acvFrm.pack(ipadx=5, ipady=10)
        acvTtlLbl = tk.Label(acvFrm, bg="white", text="Save in: ")
        acvTtlLbl.pack(padx=5, pady=5, side="left")
        self.acvEnt = tk.Entry(acvFrm, width=40)
        self.acvEnt.insert(0, "Unselected")
        self.acvEnt.configure(state="disabled")
        self.acvEnt.pack(side="left", fill=tk.X)
        acvBtn = ttk.Button(acvFrm, style="blackt.TButton",
                            text="Browse", command=self.acv_open)
        acvBtn.pack(padx=5, pady=5, side="left", fill=tk.X)

        # Grid frame
        optionGridFrm = tk.LabelFrame(
            wrpFrm, bg="white", text="Option", font=('MS Sans Serif', 12))
        optionGridFrm.pack(padx=5, pady=5, ipadx=10, ipady=2, fill="both")
        # Extension
        extTtlLbl = tk.Label(optionGridFrm, bg="white", text="Extension")
        extTtlLbl.grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        extFrm = tk.Frame(optionGridFrm, bg="white")
        extFrm.grid(row=1, column=1, sticky="nw", padx=5, pady=5)
        extLbl = tk.Label(extFrm, bg="white", text="")
        extLbl.pack(side="left")
        self.extSelect = ttk.Combobox(extFrm)
        self.extSelect.configure(
            state="readonly",
            width=8,
            value=("png", "jpg"))
        self.extSelect.current(0)
        self.extSelect.pack(side="left", padx=(3, 0))

        # Run
        runBtn = ttk.Button(wrpFrm, style="run.TButton",
                            text="Run", command=self.img_run)
        runBtn.pack(
            anchor="e",
            pady=(0, 5), padx=(0, 5),
            ipadx=10, ipady=5)

        # Message list
        msgFrm = tk.LabelFrame(
            wrpFrm, bg="white", text="Log", font=('MS Sans Serif', 12))
        msgFrm.pack(padx=5, pady=5, ipadx=5, ipady=5, fill="both")
        self.msgList = tk.Listbox(
            msgFrm, height=6, width=60, fg="#666666", bd=0, font=('MS Sans Serif', 12))
        self.msgList.pack(padx=10, pady=5, anchor=tk.W, expand=1)

        # Progress bar
        self.pb = ttk.Progressbar(
            wrpFrm, value=0, maximum=100, mode="indeterminate", orient="horizontal")
        self.pb.pack(pady=(4, 0), fill="x")

    def img_add(self):
        f_conf = [("Text Files", ("jpg", "png", "jpeg"))]
        paths = tkfd.askopenfiles(filetypes=f_conf)
        for f in paths:
            self.imgList.insert("", "end", values=(
                len(self.f_path_list), f.name))
            self.f_path_list.append(f.name)
        self.msgList.insert(0, "Added {0} image(s).".format(len(paths)))

    def img_reset(self):
        for i in self.imgList.get_children():
            self.imgList.delete(i)
        self.f_path_list = []
        self.msgList.insert(0, "Reset.")

    def acv_open(self):
        self.acv_path = tkfd.askdirectory()
        self.acvEnt.configure(state="normal")
        self.acvEnt.delete(0, "end")
        if self.acv_path != "":
            self.acvEnt.insert(0, self.acv_path)
            self.msgList.insert(0, "Selected {0}.".format(self.acv_path))
        else:
            self.acvEnt.insert(0, "unselected")
        self.acvEnt.configure(state="disabled")

    def img_run(self):
        self.pb.start(interval=10)
        # Validation
        if len(self.f_path_list) == 0:
            mess.showwarning("Warning", "Select images to check.")
            self.msgList.insert(0, "Warning: Select images to check.")
            return None
        if not hasattr(self, "acv_path"):
            mess.showwarning(
                "Warning", "Select the folder where you would like to save results.")
            self.msgList.insert(
                0, "Warning: Select the folder where you would like to save results.")
            return None
        try:
            date = datetime.datetime.now()
            self.msgList.insert(0, "[{}] Started.".format(date))
            # date = date.strftime('%Y-%m-%d-%H%M%S')
            self.result_path = os.path.join(self.acv_path, 'result')
            os.makedirs(self.result_path, exist_ok=True)
            detector_cl = Clipping()
            detector_cm = CopyMove(min_kp=20, min_match=20, min_key_ratio=0.75)
            suspicious_images = [SuspiciousImage(
                path, hist_eq=True, algorithm='orb', nfeatures=2000, gap=32) for path in self.f_path_list]
            len_sus = len(suspicious_images)
            report = pd.DataFrame(
                0,
                index=[
                    img.name for img in suspicious_images],
                columns=[
                    'Clipping',
                    'area_ratio',
                    'CopyMove',
                    'mask_ratio',
                    'Duplication', ])

            for i in range(len_sus):
                img = suspicious_images[i]
                imgname = img.name

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

            plot_pdfpages(suspicious_images, date, self.result_path)

            report.to_csv(os.path.join(self.result_path, 'report.csv'))
            self.msgList.insert(0, "[{}] Done.".format(
                datetime.datetime.now()))
        except:
            self.msgList.insert(
                0, "[{}] Failed.".format(datetime.datetime.now()))
        finally:
            self.pb.stop()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("IMDetector")
    root.geometry("650x550")
    root.option_add('*font', ('MS Sans Serif', 16))
    root.option_add('*foreground', "black")
    ttk.Style().configure('blackt.TButton', foreground='black',
                          background='white', font=('MS Sans Serif', 16))
    ttk.Style().configure('run.TButton', foreground='black',
                          background='white', font=('MS Sans Serif', 16, 'bold'))
    app = AppForm(master=root)
    app.mainloop()
