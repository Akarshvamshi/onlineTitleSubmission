from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import AcceptedTitle
from .title_processor import TitleProcessor

def homepage(request):
    if request.method == "POST":
        # Get form data from POST request
        title_name = request.POST.get("title_name")
        hindi_title = request.POST.get("hindi_title")
        owner_name = request.POST.get("owner_name")
        state = request.POST.get("state")
        publication_city_district = request.POST.get("publication_city_district")
        periodity = request.POST.get("periodity")



        # Get existing titles from database
        existing_titles = list(AcceptedTitle.objects.values_list('title_name', flat=True))

        # Process the title
        processor = TitleProcessor()
        result = (processor.process_title(title_name, existing_titles))

        request.session["title_data"] = {
            "title_name": title_name,
            "hindi_title": hindi_title,
            "owner_name": owner_name,
            "state": state,
            "publication_city_district": publication_city_district,
            "periodity": periodity,
        }
        # Prepare response
        response_data = {
            "status": result["status"],
            "verification_probability": result["verification_probability"],
            "message": result.get("reason", "Title verification completed.")
        }

        if result["status"] == "rejected":
            return render(request, "output.html", response_data)

        if "warning" in result:
            response_data["warning"] = result["warning"]

        # If title is accepted, save to database

        return render(request, "output.html", response_data)


    # Render the form if GET request
    return render(request, "home.html")

def output(request):
    return render(request, "output.html")

def add_title(request):
    # Retrieve the form data from session
    title_data = request.session.get("title_data")

    if title_data:
        # Save the title data into the database
        AcceptedTitle.objects.create(
            title_name=title_data["title_name"],
            hindi_title=title_data["hindi_title"],
            owner_name=title_data["owner_name"],
            state=title_data["state"],
            publication_city_district=title_data["publication_city_district"],
            periodity=title_data["periodity"]
        )

        # Clear the session data after saving
        del request.session["title_data"]

        return redirect("homepage")  # Redirect back to homepage after saving

    return HttpResponse("No title data available for saving.", status=400)