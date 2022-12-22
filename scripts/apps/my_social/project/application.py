
from flask import (
    Flask,
    render_template,
    request,
    flash
    )

from .models import (
    db,
    User
)

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'secret-key-goes-here'    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)
    
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/login')
    def login():
        return render_template('login.html')
    
    @app.route('/login', methods=['post'])
    def login_post():

        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        app.logger.info(f"user {email} will be logged in, or not")

        if password != '13579':
            flash('Try 13579.')
            return render_template('login.html')

        return render_template('home.html')

    @app.route('/signup')
    def signup():
        return render_template('signup.html')

    @app.route('/signup', methods=['POST'])
    def signup_post():

        email = request.form.get('email')
        password = request.form.get('password')


        flash('Signup will work later.')
        return render_template('signup.html')


    return app
