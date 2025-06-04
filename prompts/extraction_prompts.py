
JOB_DETAILS_EXTRACTOR = """
<task>
Identify the key details from a job description and company overview to create a structured JSON output. Focus on extracting the most crucial and concise information that would be most relevant for tailoring a resume to this specific job.
</task>

<job_description>
{{ job_description }}
</job_description>

Note: The "keywords", "job_duties_and_responsibilities", and "required_qualifications" sections are particularly important for resume tailoring. Ensure these are as comprehensive and accurate as possible.

{{ format_instructions }}
"""

COVER_LETTER_GENERATOR = """
<task>
Write a concise, compelling cover letter tailored to the job description and company values, using only the information provided in <my_work_information>. 
Do not invent or assume any personal stories, passions, or motivations that are not explicitly present in the input.
If a section (such as a personal hook or value alignment) cannot be written truthfully from the data, omit it or use a neutral, factual statement.
</task>

<job_description>
{{ job_description }}
</job_description>

<my_work_information>
{{ my_work_information }}
</my_work_information>

{% if application_specific_instructions %}
<application_specific_instructions>
{{ application_specific_instructions }}
</application_specific_instructions>
{% endif %}

<guidelines>
- Highlight my unique qualifications for this specific role and company culture
- Focus on the value I can bring to the employer
- Only use facts, achievements, and motivations that are explicitly present in <my_work_information>.
- Do not hallucinate or invent any details, stories, or personal connections.
- If the data lacks a personal hook or value alignment, use a neutral opening or omit that section.
- Quantify achievements and impact where possible, but only if supported by the data.
- Reference specific company projects, values, or products only if they are mentioned in <job_description> or <my_work_information>.
- Maintain a professional, authentic tone.
- If information is missing for a section, skip it rather than inventing content.
- Stay under 300 words.
</guidelines>
"""

RESUME_DETAILS_EXTRACTOR = """<objective>
Parse a text-formatted resume efficiently and extract diverse applicant's data into a structured JSON format.
</objective>

<input>
The following text is the applicant's resume in plain text format:

{{ resume_text }}
</input>

<instructions>
Follow these steps to extract and structure the resume information:

1. Analyze Structure:
   - Examine the text-formatted resume to identify key sections (e.g., personal information, education, experience, skills, certifications).
   - Note any unique formatting or organization within the resume.

2. Extract Information:
   - Systematically parse each section, extracting relevant details.
   - Pay attention to dates, titles, organizations, and descriptions.

3. Handle Variations:
   - Account for different resume styles, formats, and section orders.
   - Adapt the extraction process to accurately capture data from various layouts.

5. Optimize Output:
   - Handle missing or incomplete information appropriately (use null values or empty arrays/objects as needed).
   - Standardize date formats, if applicable.

6. Validate:
   - Review the extracted data for consistency and completeness.
   - Ensure all required fields are populated if the information is available in the resume.
</instructions>

{{ format_instructions }}"""