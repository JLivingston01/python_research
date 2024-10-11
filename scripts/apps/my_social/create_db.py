
from project import (
    create_app,
    models
    )
from project.models import db

def main():

    app = create_app()

    with app.app_context():
        db.create_all()
        
    return

if __name__=='__main__':
    main()