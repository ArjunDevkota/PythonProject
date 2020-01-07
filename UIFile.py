from tkinter import *
from featureCheck import *
from selfRF import *
import numpy as np
from ModelCreation import *

def feature_check():
    txt.delete(0.0,"end")
    url1=url.get()
    result=uiFile(url1)
    list=[]
    for i in result:
        txt.insert(END,i +'\n'+'\n')
        if "Not-Phishing" in i:
            list.append(-1)
        elif "Suspecious" in i:
            list.append(0)
        else:
            list.append(1)
    l1=np.array(list)
    l1=l1.reshape(1,-1)
    print(l1)
    finalresult=modelpredict(l1)
    txt.insert(END,"RESULT----------->"+finalresult)






#creating the object
root=Tk()

#Geometry and min and max size
root.geometry("600x400")

#title of software
root.title("Detecting Phishing Websites")

#labels and textfield
#Label for heading
labelheading=Label(root,text="Detecting Phishing Websites",font=("Times",25))
labelheading.pack(pady=10,fill=X)

#Label for URL
labelurl=Label(root,text="Enter the URL",font=("Times",15))
labelurl.pack(fill=X)

#for textfield
urltext=StringVar()
url=Entry(root,textvariable=urltext,font=('Ubuntu',16),width=25)
url.pack()

#for Button
button1=Button(root,text="CLICK HERE",bg='green',width=10,command=feature_check)
button1.pack(pady=5)

#for display result
txt=Text(root,height=30,width=150)
txt.pack()

#mainloop
root.mainloop()