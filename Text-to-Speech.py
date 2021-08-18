"""Project Description:

1.This is a Text-to-Speech-conversion project using various python libraries
A local file on the machine is given as input

2.The libraries used are
gTTS---google Text to Speech API ---input file format is .txt
pyttsx3---python text to speech library---input file is PDF format

3.Additionaly,
os module is used to open the local machine files
PyPDF2 to open the local PDF files on the machine"""

#import the file into the working environment and read after opening it

#open the .txt file from saved locally in the machine by specifying the path
op=open('C:/Users/R.R.10.02.21/Downloads/Venkata.txt','r')
#removing the newline and replacing it with a space in the .txt file for continous reading
txt=op.read().replace("\n"," ")

"""Method 1"""
from gtts import gTTS
import os

"""using gTTS directly when the file is a .txt format"""

#open the .txt file from saved locally in the machine by specifying the path
op=open('C:/Users/R.R.10.02.21/Downloads/Venkata.txt','r')
#removing the newline and replacing it with a space in the .txt file for continous reading
txt=op.read().replace("\n"," ")
language = 'en'
""" (text=input file to be read)--(lang=language of the specified file)
     (tld =accent of the language)--(slow=False--reading speed)"""
kvr = gTTS(text=txt, lang=language,tld='co.in',slow=False)
#the audio file is saved with the name "Success.mp3"
kvr.save("Success.mp3")
#the os.system opens and plays the sucess.mp3 audio file after ssucessfully saving it
os.system("Success.mp3")


"""Using pyttsx3---using the python text to speech library"""
import pyttsx3
# if the file is in PDF format
import PyPDF2
# if the file is in PDF format
pdf=open('C:/Users/R.R.10.02.21/Downloads/Venkata.pdf','rb')
read=PyPDF2.PdfFileReader(pdf)
pg=read.getPage(0)
txt=pg.extractText()
#intialise spkr
spkr=pyttsx3.init()
#let the spkr read the input file using spkr.say()
spkr.say(txt)
#let the spkr run the input file using spkr.runAndWait()
spkr.runAndWait()
