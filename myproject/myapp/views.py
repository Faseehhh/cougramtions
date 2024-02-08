from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.views import View
from .models import *
from .forms import PredictionForm
import joblib
import numpy as np
import pandas as pd
import openai
from django.conf import settings



# ----- WEASYPRINT PDF -----
from django.http import HttpResponse
from django.template import loader
from weasyprint import HTML
import datetime



def index(request):
    return render(request, "index.html")


def Courses(request):
    return render(request, "courses.html")


def Recommend(request):
    messages = []

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Get input values
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            sex = form.cleaned_data['sex']
            cet = form.cleaned_data['cet']
            gpa = form.cleaned_data['gpa']
            strand = form.cleaned_data['strand']

            # Map strand to text
            strand_mapping = {

                '0': 'Accountancy, Business, and Management (ABM)',
                '1': 'General Academic Strand (GAS)',
                '2': 'Humanities and Social Sciences (HUMSS)',
                '3': 'Sports Track (SP)',
                '4': 'Science, Technology, Engineering, and Mathematics (STEM)',
                '5': 'Technology Vocational and Livelihood (TVL)',

            }
            strand_text = strand_mapping.get(strand)

            # Prepare input data for prediction
            input_data = [[float(cet), float(gpa), float(strand), 0.0, 0.0]]


            # List of model paths
            model_paths = [
            {"path": r"C:\Users\acer\Desktop\thesis\myproject\model_aligned.pkl"},
            {"path": r"C:\Users\acer\Desktop\thesis\myproject\model_not_aligned.pkl"},
            {"path": r"C:\Users\acer\Desktop\thesis\myproject\model_mixed.pkl"},
        ]

        all_probabilities = []

        for model_info in model_paths:
            # Correctly extract the path string from the dictionary
            model_path = model_info["path"]
            
            # Load your model using the extracted path
            model = joblib.load(model_path) 
       
            if hasattr(model, 'decision_function'):
                # For models with decision_function
                decision_function_scores = model.decision_function(input_data)
                probabilities = np.exp(decision_function_scores) / np.sum(np.exp(decision_function_scores), axis=1, keepdims=True)
            else:
                # For models that provide probabilities directly
                probabilities = model.predict_proba(input_data)
            
            all_probabilities.append(probabilities)
           # Convert decision_function_scores to probabilities if they're not already
            # This step is necessary only if decision_function_scores come from a decision_function
            probabilities = np.exp(decision_function_scores) / np.sum(np.exp(decision_function_scores), axis=1, keepdims=True)

            # Get the indices of the top 3 predictions from highest to lowest
            top_3_indices = np.argsort(probabilities[0])[-3:][::-1]

            # Get class labels for the top 3 predictions
            top_3_predicted_classes = model.classes_[top_3_indices]

            # Extract the probabilities for the top 3 predicted classes
            top_3_percentages = probabilities[0, top_3_indices]

            course_mapping = {
                0: "COLLEGE OF ARCHITECTURE",
                1: "COLLEGE OF ASIAN STUDIES",
                2: "COLLEGE OF COMPUTING STUDIES",
                3: "COLLEGE OF CRIMINOLOGY",
                4: "COLLEGE OF TEACHER EDUCATION",
                5: "COLLEGE OF ENGINEERING",
                6: "COLLEGE OF HOME ECONOMICS",
                7: "COLLEGE OF LIBERAL ARTS",
                8: "COLLEGE OF NURSING",
                9: "COLLEGE OF PUBLIC ADMINISTRATION",
                10: "COLLEGE OF SCIENCE AND MATHEMATICS",
                11: "COLLEGE OF SOCIAL WORK AND COMMUNITY DEVELOPMENT",
                12: "COLLEGE OF SPORTS SCIENCE AND PHYSICAL EDUCATION",

            }


            # Determine the label for the course with the highest percentage
            highest_percentage_index = np.argmax(top_3_percentages)
            labels = [
                f"Highly Recommended! ({int(percentage * 100)}%)" if i == highest_percentage_index
                else f"({int(percentage * 100)}%)"
                for i, percentage in enumerate(top_3_percentages)
            ]

            # Displaying the cet, gpa, and strand on the first recommendation
            top_3_predicted_courses_with_description = [
                (course_mapping[course], f"Your CET {cet}, GPA {gpa}, and SHS Strand is {strand_text}", label)
                for course, label in zip(top_3_predicted_classes, labels)
            ]

            # Convert numerical indices of the top 3 predicted courses to their corresponding collges names.
            top_3_predicted_classes = [course_mapping[course_num] for course_num in top_3_predicted_classes]

            # Concatenate course names from top_3_predicted_classes into a single string separated by '|'.
            course_container = ""
            for course in top_3_predicted_classes:
                course_container += course + '|'

            # Split the concatenated string back into a list of courses.
            courses = course_container[:-1].strip().split('|')
            

            # Saving the PredResults to database
            pred_result = PredResults(
                first_name=first_name,
                last_name=last_name,
                sex=sex,
                cet=cet,
                gpa=gpa,
                # cet_status=cet_status,
                # gpa_status=gpa_status,
                strand=strand_text,
                recommended_course=courses,
            )
            pred_result.save()


            # Use OpenAI GPT-3 to generate analysis
            openai.api_key = settings.OPENAI_API_KEY

            analyses = []
            for i, course in enumerate(top_3_predicted_classes):
                if i == 0:  # First recommended college (aligned)
                    prompt_for_course = f"Base on your gpa {gpa}, cet {cet}, and strand {strand_text}, {course} aligns well with your strand. Provide an analysis that this college has a pattern with the academic background  with those of students currently enrolled in this college. Include information about the alignment with your CET scores and GPA. Additionally, explain why this college would be suitable for you. Make it in a 2-3 sentences only, and add this in the last sentence For more info regarding the available courses you can visit our home page. Provide it like you're talking to me. Use 'you or your'."

                elif i == 1:  # Second recommended college (not aligned)
                    prompt_for_course = f"Base on your gpa and cet {course} is not aligned with your strand. Say that this college has pattern with the academic background. include information about the patterns in your CET scores and GPA only and strand is not aligned. Additionally, explain why this college would be suitable for you. Dont say it for me. Make it in a 2-3 sentences only, and add this in the last sentence For more info regarding the available courses you can visit our home page. Provide it like you're talking to me. Use 'you or your'."

                elif i == 2:  # Third recommended college (mixed)
                    prompt_for_course = f"Base on your gpa, cet, and strand, {course} has a mixed alignment with your academic background. Provide an analysis that this college has a mixed pattern with the academic background. Include information about the variations in your CET scores and GPA. Additionally, explain why this college would be suitable for you. Dont say it for me. Make it in a 2-3 sentences only, and add this in the last sentence For more info regarding the available courses you can visit our home page. Provide it like you're talking to me. Use 'you or your'."

                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt_for_course,
                    max_tokens=150,
                    temperature=0.5,
                )
                

                # Append each analysis inside the loop
                analyses.append(response['choices'][0]['text'])
                print(prompt_for_course)

            # Retrieve the index of the course in predicted classes and get its corresponding percentage
            course_index = top_3_predicted_classes.index(course)
            percentage = top_3_predicted_courses_with_description[course_index][2]

            # Saving the Recommended colloges to the database
            for (course, description, percentage), analysis in zip(top_3_predicted_courses_with_description, analyses):
                recommended_course = RecommendedCourse(
                    prediction_id=pred_result,
                    course=course,
                    percentage=percentage,
                    description=description,
                    analysis=analysis,  
                )
                recommended_course.save()

            # Pass the variables to the template
            return render(request, 'result.html', {
            'analysis': analyses,
            'recommended_courses_with_description': top_3_predicted_courses_with_description,
            'first_name': first_name,
            'last_name': last_name,
            'sex': sex,
            'cet': cet,
            'gpa': gpa,
            'strand': strand_text,
            'prediction_id': pred_result.id,
            'title': 'Result',
            'messages': messages,
        })

    else:
        form = PredictionForm()

    return render(request, 'recommend.html', {'form': form, 'title': 'Recommend'})



def pdf(request, id):

    current_date = datetime.datetime.now().strftime('%B %d, %Y')

    # Assuming your HTML file is stored in the 'templates' directory
    template_path = 'pdf_template.html'

    # Get the required data from the database or wherever it's stored
    prediction = PredResults.objects.get(id=id)
    recommended = RecommendedCourse.objects.filter(prediction_id=prediction)

    # Render the template with context data if needed
    context = {'prediction': prediction, 'recommendeds': recommended, 'current_date': current_date}

    # Create a WeasyPrint HTML object
    html = HTML(string=render(request, template_path, context).content)

    # Generate PDF
    pdf_file = html.write_pdf()

    # Create a Django HttpResponse with the PDF content
    response = HttpResponse(pdf_file, content_type='application/pdf')

    # Optionally, you can set the Content-Disposition header to force download
    response['Content-Disposition'] = 'filename="output.pdf"'

    return response
