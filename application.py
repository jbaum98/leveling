from tempfile import NamedTemporaryFile

import openpyxl
import xlrd
from flask import Flask, Response, flash, redirect, render_template, request, send_file

from guarding import check_preferences_for_input_errors, output_errors
from Helpers import create_activities, create_campers, sort_campers, update_campers
from xls_output import output_master_excel

# ======================================================================================

"""This is code for creating a web-based app, mostly taken from CS50 final project"""
# Configure application
app = Flask(__name__)

# Ensure responses aren't cached
if app.config["DEBUG"]:

    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response


# Create homescreen
@app.route("/")
def index():
    """Display homescreen"""
    # User reached route via POST (as by submitting a form via POST)
    return render_template("index.html")


@app.route("/sorted", methods=["POST"])
def sorted():
    """Sort campers and download the results as an Excel document"""

    # Open the input file
    campers_location = request.files["preferences"]
    wb = xlrd.open_workbook(file_contents=campers_location.read())
    sheet = wb.sheet_by_index(0)
    # Check input file for proper input
    errors = check_preferences_for_input_errors(sheet)
    if errors != []:
        wb = output_errors(errors)
        # Download errors file
        with NamedTemporaryFile() as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            stream = tmp.read()

            r = Response(
                response=stream,
                status=200,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            r.headers["Content-Disposition"] = 'attachment; filename="errors.xlsx"'
            return r

    # Initializes the list
    campers = list()
    create_campers(campers, sheet)

    # Create activity objects
    activities_location = request.files["activities"]
    # Give the location of the input file
    activities = list()
    # Initializes the list
    create_activities(activities, activities_location)

    # Update camper objects
    if len(request.files) > 2:
        history_location = request.files["histories"]
        # Give the location of the input file
        update_campers(campers, history_location)
        # Update camper objects

    # Sort campers
    sort_campers(campers, activities)

    wb = output_master_excel(campers, activities)

    with NamedTemporaryFile() as tmp:
        wb.save(tmp.name)
        tmp.seek(0)
        stream = tmp.read()

        r = Response(
            response=stream,
            status=200,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        r.headers[
            "Content-Disposition"
        ] = 'attachment; filename="%s.xlsx"' % request.form.get("filename")
        return r
