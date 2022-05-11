from django.shortcuts import render
import numpy as np
import pandas as pd
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response


# Create your views here.

class Chatbot(APIView):

    def post(self, request):
        # Get the question from the user (request)
        data = request.data
        question = 'answer_question: ' + data['Question'].capitalize() # Capitalize to set a majuscule for the first letter

        # Load the model
        simpleT5 = ApiConfig.model

        # Get all predictions of the model
        answers = simpleT5.predict(question, num_return_sequences=5, num_beams=20)
       
        # Select an answer 
        answer = selectAnswer(request, answers)

        print('\n', request, '\n')



        # Save the QA in dict session
        saveQAInSession(request, question, answer)
        print("Les questions de l'utilisateur: ", request.session['questions'])
        print("Les réponses du bot: ", request.session['answers'])

        # Return the prediction
        response_dict = {"Bot": answer.capitalize()}
        return Response(response_dict, status=200)

def saveQAInSession(request, question, answer):
    """
        Function that stores the questions of the user and the answers and their frequencies each time.
        If the question (resp answer) has been said already in the session we increment the number of frequencies.
    """
    # We test if there is Q/A already stored for the session
    # Remark : Questions and answers are added simultaneously, we can check only one of the dict 
    #          'questions' or 'answers' not both. 

    if (not 'questions' in request.session or not request.session['questions']):
        dict_questions, dict_answers = {}, {}
        
        # Set the frequence of the question and answer to 1
        dict_questions[question] = 1
        dict_answers[answer] = 1

        # Store the QA in the appropriate dict
        request.session['questions'] = dict_questions
        request.session['answers'] = dict_answers
    
    else:
        dict_questions = request.session['questions']
        dict_answers = request.session['answers']

        # Add the question to the dict if it doesn't exist or increment the freq
        if question in dict_questions:
            dict_questions[question] += 1
        else:
            dict_questions[question] = 1

        # Add the answer to the dict if it doesn't exist or increment the freq
        if answer in dict_answers:
            dict_answers[answer] += 1
        else:
            dict_answers[answer] = 1

        request.session['questions'] = dict_questions
        request.session['answers'] = dict_answers

        request.session.modified = True

def selectAnswer(request, answers):
    """
        Function that selects the answer with the lowest frequency from the answers proposed.
    """
    dict_answers = {}

    # List of frequencies
    frequencies = []

    # We get the answers we have in the session
    if 'answers' in request.session:
        dict_answers = request.session['answers']

    # For each answer, we associate a frequency
    for ans in answers:
        if ans in dict_answers: # If the answer was already in the discussion, we get the frequency associated
            frequencies.append(dict_answers[ans])
        else:   # else, we set the frequency at 0
            frequencies.append(0)

    # The lowest frequency
    minimum_freq = np.min(frequencies)

    # We get the indices of the answers that have the lowest frequency
    indices_minimum = [i for i, freq in enumerate(frequencies) if freq == minimum_freq]
    
    # We chose randmly an answer from these answers
    index_choice = np.random.choice(indices_minimum)
    answer = answers[index_choice]
    return answer
        
        

