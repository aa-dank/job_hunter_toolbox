from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class PersonalDetails(BaseModel):
    name: str = Field(description="The full name of the candidate.")
    phone: Optional[str] = Field(default=None, description="The contact phone number of the candidate.")
    email: Optional[str] = Field(default=None, description="The contact email address of the candidate.")
    linkedin: Optional[HttpUrl] = Field(default=None, description="LinkedIn profile URL")
    github: Optional[HttpUrl] = Field(default=None, description="GitHub profile URL")

class Achievements(BaseModel):
    achievements: List[str] = Field(description="job relevant key accomplishments, awards, or recognitions that demonstrate your skills and abilities.")

class Certification(BaseModel):
    name: str = Field(description="The name of the certification.")
    by: str = Field(description="The organization or institution that issued the certification.")
    link: str = Field(description="A link to verify the certification.")

class Certifications(BaseModel):
    certifications: List[Certification] = Field(default_factory=list, description="job relevant certifications that you have earned, including the name, issuing organization, and a link to verify the certification.")

class Education(BaseModel):
    degree: str = Field(description="The degree or qualification obtained and The major or field of study. e.g., Bachelor of Science in Computer Science.")
    university: str = Field(description="The name of the institution where the degree was obtained with location. e.g. Arizona State University, Tempe, USA")
    from_date: str = Field(description="The start date of the education period. e.g., Aug 2023")
    to_date: str = Field(description="The end date of the education period. e.g., May 2025")
    courses: List[str] = Field(default_factory=list, description="Relevant courses or subjects studied during the education period. e.g. [Data Structures, Algorithms, Machine Learning]")

class Educations(BaseModel):
    education: List[Education] = Field(default_factory=list, description="Educational qualifications, including degree, institution, dates, and relevant courses.")

class Link(BaseModel):
    name: str = Field(description="The name or title of the link.")
    link: str = Field(description="The URL of the link.")

class Project(BaseModel):
    name: str = Field(description="The name or title of the project.") # Added description
    type: str | None = Field(default=None, description="The type of project, e.g., Personal, Academic, Professional.") # Added description and default
    link: str = Field(description="A link to the project repository or demo.")
    resources: Optional[List[Link]] = Field(default_factory=list, description="Additional resources related to the project, such as documentation, slides, or videos.")
    from_date: str = Field(description="The start date of the project. e.g., Aug 2023") # Added description
    to_date: str = Field(description="The end date of the project. e.g., Nov 2025 or Present") # Added description
    description: List[str] = Field(default_factory=list, description="A list of 3 bullet points describing the project experience, tailored to match job requirements. Each bullet point should follow the 'Did X by doing Y, achieved Z' format, quantify impact, implicitly use STAR methodology, use strong action verbs, and be highly relevant to the specific job. Ensure clarity, active voice, and impeccable grammar.")

class Projects(BaseModel):
    projects: List[Project] = Field(default_factory=list, description="Project experiences, including project name, type, link, resources, dates, and description.")

class SkillSection(BaseModel):
    name: str = Field(description="name or title of the skill group and competencies relevant to the job, such as programming languages, data science, tools & technologies, cloud & DevOps, full stack,  or soft skills.")
    skills: List[str] = Field(default_factory=list, description="Specific skills or competencies within the skill group, such as Python, JavaScript, C#, SQL in programming languages.")

class SkillSections(BaseModel):
    skill_section: List[SkillSection] = Field(default_factory=list, description="Skill sections, each containing a group of skills and competencies relevant to the job.")

class Experience(BaseModel):
    role: str = Field(description="The job title or position held. e.g. Software Engineer, Machine Learning Engineer.")
    company: str = Field(description="The name of the company or organization.")
    location: str = Field(description="The location of the company or organization. e.g. San Francisco, USA.")
    from_date: str = Field(description="The start date of the employment period. e.g., Aug 2023")
    to_date: str = Field(description="The end date of the employment period. e.g., Nov 2025")
    description: List[str] = Field(default_factory=list, description="A list of 3 bullet points describing the work experience, tailored to match job requirements. Each bullet point should follow the 'Did X by doing Y, achieved Z' format, quantify impact, implicitly use STAR methodology, use strong action verbs, and be highly relevant to the specific job. Ensure clarity, active voice, and impeccable grammar.")

class Experiences(BaseModel):
    work_experience: List[Experience] = Field(default_factory=list, description="Work experiences, including job title, company, location, dates, and description.")

class Media(BaseModel):
    linkedin: Optional[HttpUrl] = Field(default=None, description="LinkedIn profile URL") # Added default
    github: Optional[HttpUrl] = Field(default=None, description="GitHub profile URL") # Added default
    medium: Optional[HttpUrl] = Field(default=None, description="Medium profile URL") # Added default


class ResumeSchema(BaseModel):
    personal: PersonalDetails = Field(description="Personal contact information and professional media links for the candidate.")
    summary: Optional[str] = Field(default=None, description="A brief summary of the candidate's profile and career objectives, tailored to the job.")
    experiences: Experiences = Field(default_factory=Experiences)
    projects: Projects = Field(default_factory=Projects)
    educations: Educations = Field(default_factory=Educations)
    skills: SkillSections = Field(default_factory=SkillSections)
    achievements: Optional[Achievements] = Field(default=None)
    certifications: Optional[Certifications] = Field(default=None)
    

class JobDetails(BaseModel):
    job_title: str = Field(description="The specific role, its level, and scope within the organization.")
    job_purpose: str = Field(description="A high-level overview of the role and why it exists in the organization.")
    keywords: List[str] = Field(description="Key expertise, skills, and requirements the job demands.")
    job_duties_and_responsibilities: List[str] = Field(description="Focus on essential functions, their frequency and importance, level of decision-making, areas of accountability, and any supervisory responsibilities.")
    required_qualifications: List[str] = Field(description="Including education, minimum experience, specific knowledge, skills, abilities, and any required licenses or certifications.")
    preferred_qualifications: List[str] = Field(description="Additional \"nice-to-have\" qualifications that could set a candidate apart.")
    company_name: str = Field(description="The name of the hiring organization.")
    company_details: str = Field(description="Overview, mission, values, or way of working that could be relevant for tailoring a resume or cover letter.")