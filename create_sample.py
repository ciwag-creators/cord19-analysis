import pandas as pd

def create_sample_data():
    # Create a small sample dataset for demo
    sample_data = {
        'cord_uid': ['abc123', 'def456', 'ghi789'],
        'title': [
            'COVID-19 Transmission Dynamics',
            'Vaccine Efficacy Study',
            'Public Health Response Analysis'
        ],
        'authors': ['Smith J et al.', 'Johnson A et al.', 'Lee K et al.'],
        'journal': ['Lancet', 'Nature', 'Science'],
        'publish_time': ['2020-03-15', '2021-02-20', '2022-01-10'],
        'abstract': [
            'Study of transmission patterns in urban areas.',
            'Analysis of vaccine effectiveness across populations.',
            'Evaluation of public health measures during pandemic.'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_metadata.csv', index=False)
    print("Sample dataset created: sample_metadata.csv")

if __name__ == "__main__":
    create_sample_data()