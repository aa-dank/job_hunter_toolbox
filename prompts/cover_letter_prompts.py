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