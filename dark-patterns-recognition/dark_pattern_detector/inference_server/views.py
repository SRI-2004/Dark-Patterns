# Create your views here.
import json
from django.db.models import Count
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from joblib import load
import torch

from .models import DetectionResult

presence_tokenizer = load('/home/srinivasan/Desktop/Dark-Patterns/dark-patterns-recognition/dark_pattern_detector/inference_server/train_classifier/models/presence_tokenizer_distilbert.joblib')
presence_model = load('/home/srinivasan/Desktop/Dark-Patterns/dark-patterns-recognition/dark_pattern_detector/inference_server/train_classifier/models/presence_classifier_distilbert.joblib')
category_model = load('/home/srinivasan/Desktop/Dark-Patterns/dark-patterns-recognition/dark_pattern_detector/inference_server/train_classifier/models/category_classifier_bert.joblib')
category_tokenizer = load('/home/srinivasan/Desktop/Dark-Patterns/dark-patterns-recognition/dark_pattern_detector/inference_server/train_classifier/models/category_tokenizer_bert.joblib')
device = "cuda" if torch.cuda.is_available() else "cpu"


classes = ["Social Proof","Misdirection","Urgency","Forced Action","Obstruction","Sneaking","Scarcity"]

@csrf_exempt
def main(request: HttpRequest):
    if request.method == 'POST':
        output = []
        site_url = json.loads(request.body.decode('utf-8')).get('site_url')
        data = json.loads(request.body.decode('utf-8')).get('tokens')

        for token in data:

            token_lower = token.lower()
            # if any(word in token_lower for word in blacklist):
            #     continue
            input_ids = presence_tokenizer.encode(token, return_tensors='pt')
            attention_mask = torch.ones_like(input_ids)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = presence_model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                confidence_score = torch.nn.functional.softmax(outputs, dim=1)[:, 1].item()

            # Set a confidence threshold for presence classification
            presence_confidence_threshold = 0

            if predictions == 1 and confidence_score >= presence_confidence_threshold:
                result = 'Dark'
            else:
                result = 'Not Dark'  # Not Dark

            if result == 'Dark':

                input_encoding = category_tokenizer(token, truncation=True, padding=True, return_tensors='pt')
                input_encoding = input_encoding.to(device)

                with torch.no_grad():
                    logits = category_model(**input_encoding).squeeze()

                    # Apply softmax to get probabilities
                    probabilities = torch.nn.functional.softmax(logits, dim=0)

                    # Get the predicted class and its corresponding confidence score
                    predicted_class = torch.argmax(probabilities).item()
                    confidence_score = probabilities[predicted_class].item()
                    # print(token, classes[predicted_class], confidence_score)
                    # Set a confidence threshold for category classification
                    category_confidence_threshold = 0
                    class_value = classes[predicted_class]
                    # Check if confidence exceeds the threshold
                    if confidence_score >= category_confidence_threshold:
                        output.append(class_value)

                    detection_result = DetectionResult(
                        site_url=site_url,
                        phrase=token[:1000],
                        dark_pattern_type=class_value,
                        confidence_score=confidence_score
                    )
                    detection_result.save()

            else:
                output.append(result)

        dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
        # for d in dark:
        #     print(d)
        # print()
        # print(len(dark))

        message = '{ \'result\': ' + str(output) + ' }'
        # print(message)
        print(message)
        # response_data = {'result': str(output)}
        return JsonResponse(message,safe=False)


    return JsonResponse({'error': 'Invalid request method'}, status=405)
@csrf_exempt
def feedback(request: HttpRequest):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            feedback = data.get('feedback')
            dark_pattern = data.get('pattern')
            site_url = data.get('site_url')

            detection_result = DetectionResult(
                site_url=site_url,
                phrase=feedback,
                dark_pattern_type=dark_pattern,
                confidence_score=1,
                feedback=True
            )
            detection_result.save()

            # Save the changes to the database
            detection_result.save()
            # Printing feedback and dark pattern to terminal
            print("site_url:", site_url)
            print("Feedback:", feedback)
            print("Dark Pattern:", dark_pattern)

            return JsonResponse({'message': 'Feedback received successfully'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data in request body'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def get_detection_results(request: HttpRequest):
    if request.method == 'GET':
        # Get all unique site URLs from the DetectionResult model
        site_urls = DetectionResult.objects.values_list('site_url', flat=True).distinct()

        # Initialize a dictionary to store the data for each site URL
        data = {}

        for site_url in site_urls:
            detection_results = DetectionResult.objects.filter(site_url=site_url)

            # Initialize a dictionary to store the count and example phrases for each dark pattern type
            site_data = {class_name: {'count': 0, 'examples': []} for class_name in classes}

            for result in detection_results:
                site_data[result.dark_pattern_type]['count'] += 1
                if len(site_data[result.dark_pattern_type]['examples']) < 5:
                    site_data[result.dark_pattern_type]['examples'].append(result.phrase)

            # Add the data for this site URL to the main data dictionary
            data[site_url] = site_data

        # List of domains for which to calculate the overall count
        domains = ['amazon']

        for domain in domains:
            # Get all detection results where the site URL contains the domain
            detection_results = DetectionResult.objects.filter(site_url__contains=domain)

            # Initialize a dictionary to store the count for each dark pattern type
            domain_data = {class_name: {'count': 0} for class_name in classes}

            for result in detection_results:
                domain_data[result.dark_pattern_type]['count'] += 1

            # Add the overall count for this domain to the main data dictionary
            data[domain + '_overall'] = domain_data

        return JsonResponse(data)

    return JsonResponse({'error': 'Invalid request method'}, status=405)