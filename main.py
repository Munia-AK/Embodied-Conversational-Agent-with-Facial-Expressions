#!/usr/bin/env python
# coding: utf-8

# All Models 
"""
combining speech recognition, response generation, emotion recognition models
talk --> speech to text --> create response --> detect emotion --> save to files

UDP connection is not used here
we save outputs in files and global variables
outputs: emotion and response sound file
"""

# imports

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

import sys
import time, logging
from datetime import datetime
import collections, queue, os, os.path

import itertools
import threading
from threading import Thread

import pandas as pd
import numpy as np
import ktrain
from ktrain import text

import random
import sched

import argparse
import shutil, os

## imports for parlai blender chatbot

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
import parlai.utils.logging as parlaiLogging

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.utils.misc import display_messages, load_cands
from parlai.utils.strings import colorize

# imports for Speech Recognition using the Speech SDK
import azure.cognitiveservices.speech as speechsdk

# imports for synthesizing text using the Speech SDK
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

""" --------------------------------------------------------------------"""

##### load models #####

main_path = 'C:/Users/Munia/Documents/Anaconda/Master Thesis Project/vh-models/'


# load text based emotion recognition model
t_predictor_path = main_path + "emotion-recognition/emotion recognition from text/Bert/models/bert_model/"
t_predictor = ktrain.load_predictor(t_predictor_path)

# load video based emotion recognition models
v_detection_model_path = main_path + 'emotion-recognition/real time emotion detection/haarcascade_files/haarcascade_frontalface_default.xml'
v_emotion_model_path = main_path + 'emotion-recognition/real time emotion detection/models/XCEPTION-102-0.66.hdf5'
face_detection = cv2.CascadeClassifier(v_detection_model_path)
v_emotion_classifier = load_model(v_emotion_model_path, compile=False)

# load chatbot model for response generation
chatbot_model = 'open-domain-chatbot/models/blender/blender_90M/model'

""" --------------------------------------------------------------------"""

##### speech to text #####

###### speech reconition using Microsoft Speech API #####

## Speech Recognition Using the Speech SDK

def speak():
    global sentence
    speech_key, service_region = "5b49444222a544a39f55e5544466cd82", "eastasia" # must change API key and region name
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("say something: ")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason. RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        sentence = result.text
        output_file = open(main_path + "outputs/text-output.txt", "w")
        output_file.write(result.text)
        output_file.close()
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        print("assigning sentence as -> say something")
        sentence = "say something"
        output_file = open(main_path + "outputs/text-output.txt", "w")
        output_file.write(sentence)
        output_file.close()
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    
    
# loading animation
def spinner():
    
    done = False
    #here is the animation
    def animate():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if done:
                break
            sys.stdout.write('\rloading ' + c)
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\rDone!     ')

    t = threading.Thread(target=animate)
    t.start()

    #long process here
    time.sleep(10)
    done = True

""" --------------------------------------------------------------------"""

###### text to speech with Microsoft Speech API #####

def synthesize(responsetxt):
    
    #file="C:/Users/Munia/Documents/Unity/VHProject-master-Copy/Assets/Resources/speech.wav"
    file = "outputs/speech.wav"
    speech_config = SpeechConfig(subscription="5b49444222a544a39f55e5544466cd82", region="eastasia")
    #audio_config = AudioOutputConfig(filename="outputs/speech.wav")
    audio_config = AudioOutputConfig(filename=file)
    
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(responsetxt)

""" --------------------------------------------------------------------"""

##### emotion recognition #####

### from text

def text_based_emotion(text):
    global t_emotion
    global t_prob
    EMOTIONS = ['happy', 'sad', 'scared', 'angry', 'neutral']
    
    start_time = time.time() 
    preds = t_predictor.predict(text, return_proba=True)
    emotion_probability = np.max(preds)
    emotion_probability = str(round(emotion_probability, 2))
    #convert probability from string to float
    probability = float(emotion_probability)
    #print("blah blah blah: ", probability)
    label = EMOTIONS[preds.argmax()]

    #print('predicted: {} {} ({:.2f})'.format(label, emotion_probability, (time.time() - start_time)))
    t_emotion = label
    t_prob = probability
    
    #return label, emotion_probability


### from video

# video capture is set to play for 5 seconds
def video_based_emotion():
    
    #main_path = 'C:/Users/Munia/Documents/Anaconda/vh-models/emotion-recognition/real time emotion detection/'
    global v_emotion
    global v_prob
    
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

    cv2.namedWindow('your_face')
    t0 = time.time() # start time in seconds
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        #reading the frame
        frame = imutils.resize(frame,width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)


            preds = v_emotion_classifier.predict(roi)[0]
            #print(preds)
            emotion_probability = np.max(preds)
            #print(np.max(preds))
            label = EMOTIONS[preds.argmax()]
            #print(label)
        else: continue

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
        t1 = time.time() # current time
        num_seconds = t1 - t0 # diff

        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
        
        # stop video capture after x seconds
        if num_seconds > 5:  # e.g. break after 30 seconds
            break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    
    camera.release()
    cv2.destroyAllWindows()
    v_emotion = label
    v_prob = emotion_probability
    
### predict final emotion

# last function for emotion detection that comapres between emotion from video and emotion from text
def predict_emotion():
    emotionsList = ["happy", "sad", "neutral", "angry", "scared", "surprised", "disgust"]
    final_emotion = ""
    final_prob = 0
    
    if (t_emotion==v_emotion):
        final_emotion = t_emotion
        final_prob = t_prob
        return final_emotion, final_prob
    else:
        if (v_prob is not None):
            MAX = max(t_prob, v_prob) 
            if(MAX == t_prob):
                final_emotion = t_emotion
                final_prob = t_prob
                return final_emotion, final_prob
            elif(MAX == v_prob):
                final_emotion = v_emotion
                final_prob = v_prob
                return final_emotion, final_prob
        else:
            final_emotion = t_emotion
            final_prob = t_prob
            return final_emotion, final_prob

""" --------------------------------------------------------------------"""

##### generate response #####
### response generation with ParlAI Blender 90M Model

"""
LocalHumanAgent Class (for keyboard input) to let human reply replacing an ML agent
instead of importing  LocalHumanAgent.py Script

"""

"""
Agent does gets the local keyboard input in the act() function.
Example: parlai eval_model -m local_human -t babi:Task1k:1 -dt valid
"""
class LocalHumanAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Local Human Arguments')
        agent.add_argument(
            '-fixedCands',
            '--local-human-candidates-file',
            default=None,
            type=str,
            help='File of label_candidates to send to other agent',
        )
        agent.add_argument(
            '--single_turn',
            type='bool',
            default=False,
            help='If on, assumes single turn episodes.',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.finished = False
        self.fixedCands_txt = load_cands(self.opt.get('local_human_candidates_file'))
        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                'highlight',
            )
        )

    def epoch_done(self):
        return self.finished

    def observe(self, msg):
        print(
            display_messages(
                [msg],
                add_fields=self.opt.get('display_add_fields', ''),
                prettify=self.opt.get('display_prettify', False),
                verbose=self.opt.get('verbose', False),
            )
        )

    def act(self):
        reply = Message()
        reply['id'] = self.getID()
        try:
            ## input message from user
            #reply_text = input(colorize("Enter Your Message:", 'text') + ' ')
            reply_text = sentence
        except EOFError:
            self.finished = True
            return {'episode_done': True}

        #reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        if self.opt.get('single_turn', False):
            reply.force_set('episode_done', True)
        reply['label_candidates'] = self.fixedCands_txt
        reply['text'] = reply_text
        print("reply['text'] is = ", reply['text'])
        print("reply is = ", reply)
      
        return reply

    def episode_done(self):
        return self.episodeDone

"""
Basic script which allows local human keyboard input to talk to a trained model.

Model = Blender 90M
"""

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'Interactive chat with a model on the command line'
        )
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-add-fields',
        type=str,
        default='',
        help='Display these fields when verbose is off (e.g., "--display-add-fields label_candidates,beam_texts")',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.add_argument(
        '--outfile',
        type=str,
        default='',
        help='Saves a jsonl file containing all of the task examples and '
        'model replies. Set to the empty string to not save at all',
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser, partial_opt=None)
    WorldLogger.add_cmdline_args(parser, partial_opt=None)
    return parser

##### main function for chtabot to build a chat(from interactive script) 

def interactive(opt):
    global response
    if isinstance(opt, ParlaiParser):
        parlaiLogging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    #agent.opt.log()
    human_agent = LocalHumanAgent(opt)
    # set up world logger
    world_logger = WorldLogger(opt) if opt.get('outfile') else None
    world = create_task(opt, [human_agent, agent])

    # Show some example dialogs:
    world.parley()
    #world.get_acts()
    response = world.get_acts()[1]['text']
    #print(world.get_acts()[1]['text'])
    #a = world.get_acts()

@register_script('interactive', aliases=['i'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return interactive(self.opt)

##### Generate Response Function, calles the interactive function
def responseGenerate():
    random.seed(42)
    Interactive.main(model_file=chatbot_model)

""" --------------------------------------------------------------------"""

##### Main Code // run all models #####

##### function to print the resulted chat showing what user said, response, emotions

def  print_dialog():
    print("\n\n____________ Dialog ____________")
    print("Me: ", sentence)
    print("emotion from video is: ", v_emotion)
    print("probability  is: ", v_prob)

    print("Bot: ", response)
    print("emotion from text is: ", t_emotion)
    print("probability is: ", t_prob)

    print("----------------------------------------")
    print("Outputs: ")
    print("final emotion output is: ", final_emo, "with probability: ", final_prob)
    print("bot response output is: ", response)

    
""" --------------------------------------------------------------------"""

final_emo = None
t_emotion = None
t_prob = None
v_emotion = None
v_prob = None
sentence = None
response = None


# call speech recognition, with animation, and video based emotion functions together in parallel with threads
# store spoken sentence and video emotion in gloabal variables
# call generate response and store the response in gloabal variables
# call text based emotion on response
# call synthesize function to convert response text to speech and store it as file
# print the dialog
# call the predict emotion funcction to pick final emotion 
    #compare between video and text emotions and pick the one with lareger probability
    # store the emotion as file
# wait for seconds then repeat again

while True:
    
    t1 = Thread(target = speak)
    t2 = Thread(target = spinner)
    t3 = Thread(target = video_based_emotion)
    
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    
    responseGenerate()
    text_based_emotion(response)
        
    final_emo, final_prob = predict_emotion()
    synthesize(response)

    print_dialog()

    # saving emotion in outputs folder // speech has been saved in the synthesize function
    output_emotion = open(main_path + "outputs/emotion.txt", "w")
    output_emotion.write(final_emo)
    output_emotion.close()

    # saving emotion and speech in Unity's Resources folder

    directory = "C:/Users/Munia/Documents/Unity/VHProject-master-Copy/Assets/Resources/"

    output_emotion = open(directory + "emotion.txt", "w")
    output_emotion.write(final_emo)
    output_emotion.close()
    
    #copy speech file from outputs folder to Unity's resources folder
    path1 = main_path + 'outputs/speech.wav'
    path2 = directory + 'speech.wav'
    shutil.copy(path1, path2)

    t_emotion = None
    t_prob = None
    v_emotion = None
    v_prob = None
    sentence = None
    response = None
    print("wait for 10 seconds")
    i=1
    while(i<=10):
        print(i)
        i+=1
        time.sleep(2)
