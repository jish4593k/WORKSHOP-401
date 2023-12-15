from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import torch
import tqdm
import speech
import speech.loader as loader
import tkinter as tk
from tkinter import filedialog

class SpeechEvaluator:
    def __init__(self, model_path, dataset_json, use_best_model=True, batch_size=8, out_file=None):
        self.model_path = model_path
        self.dataset_json = dataset_json
        self.use_best_model = use_best_model
        self.batch_size = batch_size
        self.out_file = out_file

    def evaluate(self):
        use_cuda = torch.cuda.is_available()

        model, preproc = speech.load(self.model_path, tag=None if self.use_best_model else "best")
        ldr = loader.make_loader(self.dataset_json, preproc, self.batch_size)

        model.cuda() if use_cuda else model.cpu()
        model.set_eval()

        all_preds = []
        all_labels = []

        for batch in tqdm.tqdm(ldr):
           
            inputs = torch.tensor(batch[0]).cuda() if use_cuda else torch.tensor(batch[0])
            preds = model.infer(inputs)
            all_preds.extend(preds)
            all_labels.extend(batch[1])

        results = [(preproc.decode(label), preproc.decode(pred)) for label, pred in zip(all_labels, all_preds)]
        cer = speech.compute_cer(results)
        print("CER {:.3f}".format(cer))

        if self.out_file is not None:
            self.save_results(results)

    def save_results(self, results):
        with open(self.out_file, 'w') as fid:
            for label, pred in results:
                res = {'prediction': pred, 'label': label}
                json.dump(res, fid)
                fid.write("\n")

class SpeechGUI:
    def __init__(self, master):
        self.master = master
        master.title("Speech Model Evaluator")

        self.model_label = tk.Label(master, text="Model Path:")
        self.model_label.pack()

        self.model_entry = tk.Entry(master)
        self.model_entry.pack()

        self.dataset_label = tk.Label(master, text="Dataset Path:")
        self.dataset_label.pack()

        self.dataset_entry = tk.Entry(master)
        self.dataset_entry.pack()

        self.batch_size_label = tk.Label(master, text="Batch Size:")
        self.batch_size_label.pack()

        self.batch_size_entry = tk.Entry(master)
        self.batch_size_entry.pack()

        self.output_label = tk.Label(master, text="Output File (optional):")
        self.output_label.pack()

        self.output_entry = tk.Entry(master)
        self.output_entry.pack()

        self.evaluate_button = tk.Button(master, text="Evaluate", command=self.evaluate)
        self.evaluate_button.pack()

    def evaluate(self):
        model_path = self.model_entry.get()
        dataset_path = self.dataset_entry.get()
        batch_size = int(self.batch_size_entry.get())
        output_path = self.output_entry.get() if self.output_entry.get() else None

      
        evaluator = SpeechEvaluator(**config)
        evaluator.evaluate()


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechGUI(root)
    root.mainloop()
