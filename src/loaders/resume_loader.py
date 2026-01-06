"""Loader for resume data."""

import json
from pathlib import Path
from typing import List

from ..models.resume import Resume, ExperienceItem, EducationItem, ProjectItem


class ResumeLoader:
    """Loads resume data from files and returns domain objects."""
    
    def load_from_json(self, path: str) -> Resume:
        """
        Load resume from JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Resume domain object
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Resume file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract full text (concatenate all text fields)
        text_parts = []
        if 'name' in data:
            text_parts.append(f"Name: {data['name']}")
        if 'summary' in data:
            text_parts.append(f"Summary: {data['summary']}")
        if 'skills' in data:
            text_parts.append(f"Skills: {', '.join(data['skills'])}")
        if 'experience' in data:
            for exp in data['experience']:
                text_parts.append(f"{exp.get('role', '')} at {exp.get('company', '')}")
                text_parts.extend(exp.get('bullets', []))
        if 'education' in data:
            edu = data['education']
            text_parts.append(f"{edu.get('degree', '')} from {edu.get('school', '')}")
        if 'projects' in data:
            for proj in data['projects']:
                text_parts.append(f"{proj.get('name', '')} - {proj.get('domain', '')}")
                text_parts.extend(proj.get('bullets', []))
        if 'leadership' in data:
            text_parts.extend(data['leadership'])
        full_text = '\n'.join(text_parts)
        
        # Extract skills
        skills = data.get('skills', [])
        
        # Extract experience
        experience: List[ExperienceItem] = []
        for exp_data in data.get('experience', []):
            description = '\n'.join(exp_data.get('bullets', []))
            experience.append(ExperienceItem(
                title=exp_data.get('role', ''),
                company=exp_data.get('company', ''),
                description=description,
                start_date=exp_data.get('start_date'),
                end_date=exp_data.get('end_date'),
                location=exp_data.get('location')
            ))
        
        # Extract education
        education: List[EducationItem] = []
        if 'education' in data:
            edu_data = data['education']
            # Use graduation as end_date if start_date not provided
            graduation = edu_data.get('graduation')
            education.append(EducationItem(
                school=edu_data.get('school', ''),
                degree=edu_data.get('degree', ''),
                start_date=edu_data.get('start_date'),
                end_date=edu_data.get('end_date') or graduation,
                gpa=edu_data.get('gpa'),
                honors=edu_data.get('honors'),
                coursework=edu_data.get('coursework', [])
            ))
        
        # Extract projects
        projects: List[ProjectItem] = []
        for proj_data in data.get('projects', []):
            description = '\n'.join(proj_data.get('bullets', []))
            projects.append(ProjectItem(
                name=proj_data.get('name', ''),
                domain=proj_data.get('domain'),
                description=description,
                bullets=proj_data.get('bullets', [])
            ))
        
        # Extract extra activities (leadership, clubs, etc.)
        extra_activities = []
        if 'leadership' in data:
            extra_activities.extend(data['leadership'])
        if 'clubs' in data:
            extra_activities.extend(data['clubs'])
        if 'activities' in data:
            extra_activities.extend(data['activities'])
        
        # Metadata
        metadata = {
            'filename': path_obj.name,
            'name': data.get('name'),
            'contact': data.get('contact', {})
        }
        
        return Resume(
            text=full_text,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            extra_activities=extra_activities,
            metadata=metadata
        )

