from tkinter import ttk
import tkinter as tk
import pickle
import os
import sys
class OneShot(tk.Frame):
    def __init__(self, root = None):
        tk.Frame.__init__(self, root)
        self.master = root
        self.identities = []
        self.filename = "temp/identities.pickle"
        self.createUI()
        self.createMenu()
        self.bindEvents()
        self.master.after(2000, self.updateGUI)
    def run(self):
        print("Creating thread")
    def bindEvents(self):
        self.master.bind("<Escape>", self.exit)
    def showAbout(self):
        pass
    def updateGUI(self):
        l = pickle.loads(open(self.filename, "rb").read())
        for i in l:
            if i not in self.identities:
                self.AddToList(i)
        self.master.update()
        self.master.after(2000, self.updateGUI)
    def createMenu(self):
        mb = tk.Menu(self.master)
        fmenu = tk.Menu(mb)
        fmenu.add_command(label="About", command=lambda:self.showAbout())
        fmenu.add_command(label="Exit", command= lambda: exit())
        mb.add_cascade(label="OneShot", menu = fmenu)
        self.master.config(menu=mb)

    def createUI(self):
        self.master.title("Identities Detected")
        # self.master.geometry("400x400")
        self.ListLabel = ttk.Label(self.master, text="Identities detected:")
        self.ListBox1 = tk.Listbox(self.master)

    def AddToList(self, identity):
        if identity not in self.identities:
            self.ListBox1.insert(len(self.identities), identity)
            self.ListBox1.pack()
            self.identities.append(identity)

    def RemoveFromList(self, identity):
        if identity in self.identities:
            index = self.identities.index(identity)
            self.ListBox1.delete(index)
            self.identities[index]
    def ResetList(self):
        self.ListBox1.delete(0, tk.END)
    def exit(self, event):
        self.master.destroy()


class SharedCell():
    def __init__(self):
        self.l = []
        self.filename = "temp/identities.pickle"
        with open(self.filename, "wb") as f:
            f.write(pickle.dumps(self.l))
    def AddToList(self, identity):
        self.l = pickle.loads(open(self.filename,"rb").read())
        if identity not in self.l:
            self.l.append(identity)
            with open(self.filename, "wb") as f:
                f.write(pickle.dumps(self.l))
    def reset(self):
        os.remove(self.filename)


if __name__ == '__main__':
    root = tk.Tk()
    t = OneShot(root)
    root.mainloop()