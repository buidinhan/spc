"""
How to use the script?

1. The script is intended to work with the standard registry template
consisting of one cover page and 99 pages for routine activities. Any
changes to the template may prevent the script from working properly.
If you want to use another template, you must modify the script.

2. To make a new registry, you have to prepare by yourself a pdf file
containing the cover page, and a pdf file containing the forms for routine
activities. You can prepare them as Word documents and then convert them
to pdf. It should be easy for you. :)

3. Make sure there are 3 pdf files named "cover.pdf", "routine_log.pdf",
and "template.pdf" in the same folder as the file named "_registry_maker.py".
Remember that you shouldn't make any changes to the "template", even its
name. If you want to change anything, make sure you know how to modify the
script.

4. Run the script by double-clicking the file named "_registry_maker.py".
Wait for a while for the script to finish making the log book.

5. Move the resulted registry to another folder or it'll be overwritten when
you make another registry.

I hope the script will make your life a bit easier. ^.^
"""

import time, sys

import PyPDF2 # v1.26.0


# PREPARATION
number_of_pages = 100
# Source files
template_filename = "template.pdf"
cover_filename = "cover.pdf"
routine_log_filename = "routine_log.pdf"

# Output file
registry_filename = "REGISTRY.pdf"

# Open all pdf files.
try:
    template_file = open(template_filename, "rb")
    cover_file = open(cover_filename, "rb")
    routine_log_file = open(routine_log_filename, "rb")
    registry_file = open(registry_filename, "wb")
except FileNotFoundError:
    print("Some files are missing.")
    time.sleep(2)
    sys.exit()

# Get the pdf readers and writer.
try:
    template_reader = PyPDF2.PdfFileReader(template_file)
    cover_reader = PyPDF2.PdfFileReader(cover_file)
    routine_log_reader = PyPDF2.PdfFileReader(routine_log_file)
    registry_writer = PyPDF2.PdfFileWriter()
except:
    print("Unable to read some files.")
    time.sleep(2)
    sys.exit()


# PROCESSING PAGES
# Make the cover page.
print("Processing page 1 ...")
registry_cover_page = template_reader.getPage(0)
cover_content = cover_reader.getPage(0)
registry_cover_page.mergePage(cover_content)
registry_writer.addPage(registry_cover_page)

# Make the routine log pages.
numPages = routine_log_reader.numPages
if numPages == 1:
    # All routine log pages have the same content.
    routine_log_content = routine_log_reader.getPage(0)
    for page_index in range(1, number_of_pages):
        print("Processing page {} ...".format(page_index+1))
        routine_log_page = template_reader.getPage(page_index)
        routine_log_page.mergePage(routine_log_content)
        registry_writer.addPage(routine_log_page)
else:
    # There are different contents on adjacent pages.
    routine_log_contents = [routine_log_reader.getPage(i) for i in
                            range(numPages)]
    for page_index in range(1, number_of_pages):
        print("Processing page {} ...".format(page_index+1))
        routine_log_content = routine_log_contents[(page_index-1) % numPages]
        routine_log_page = template_reader.getPage(page_index)
        routine_log_page.mergePage(routine_log_content)
        registry_writer.addPage(routine_log_page)
    
# Write all pages to the output pdf file.
print("Finishing the registry ...")
registry_writer.write(registry_file)

# Close all pdf files.
registry_file.close()
template_file.close()
cover_file.close()
routine_log_file.close()

print("DONE.")
time.sleep(2)
