
from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for
    )

from flask_login import (
    login_required,
    LoginManager,
    login_user,
    current_user,
    logout_user
    )

from werkzeug.security import(
    generate_password_hash,
    check_password_hash
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

    login_manager = LoginManager()
    login_manager.login_view='login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
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

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password,password):
            flash('Please check your login details and try again')
            return redirect(url_for('login'))

        login_user(user, remember=remember)

        app.logger.info(f"user {user} logged in")

        return redirect(url_for('userhome'))

    @app.route('/userhome')
    @login_required
    def userhome():
        curr_name = current_user.firstname
        return render_template('userhome.html',firstname=curr_name)

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('index'))

    @app.route('/signup')
    def signup():
        return render_template('signup.html')

    @app.route('/signup', methods=['POST'])
    def signup_post():

        email = request.form.get('email')
        password = request.form.get('password')
        firstname = request.form.get('firstname')
        lastname = request.form.get('lastname')
        location = request.form.get('location')
        occupation = request.form.get('occupation')
        company = request.form.get('company')
        about = request.form.get('about')

        user = User.query.filter_by(email=email).first()

        if user:
            flash('Email address is already in use.')
            return redirect(url_for('signup'))

        new_user = User(email=email,
            password= generate_password_hash(password,method='sha256'),
            firstname=firstname,
            lastname=lastname,
            location=location,
            occupation=occupation,
            company=company,
            aboutme=about,
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    @app.route('/amiloggedin')
    @login_required
    def am_i_logged_in():

        return render_template('amiloggedin.html')


    return app
