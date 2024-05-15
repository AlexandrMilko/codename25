from disruptor import db, login_manager, app
from flask_login import UserMixin
from itsdangerous import URLSafeTimedSerializer as Serializer

@login_manager.user_loader
def load_user(id):
    return User.query.get(str(id))

class User(db.Model, UserMixin):
    # We store it as String because Integer is too small for Google's unique id
    id = db.Column(db.String(30), primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    #We set it to nullable true, because via Google authentication we do not need to specify the password. We dont have it
    password = db.Column(db.String(100), nullable=True)

    # Used for verifying email in password reset
    def get_reset_token(self, expires_sec=1800): # TODO fix expires_sec
        s = Serializer(app.config["SECRET_KEY"])
        return s.dumps({"user_id": self.id})

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config["SECRET_KEY"])
        try:
            user_id = s.loads(token)["user_id"]
        except:
            return None
        return load_user(user_id)

    # Used in verifying email in registration
    def get_email_verification_token(self, expires_sec=1800): # TODO fix expires_sec
        s = Serializer(app.config["SECRET_KEY"])
        return s.dumps({
            "user_id": self.id,
            "username": self.username,
            "email": self.email,
            "password": self.password
                        })

    @staticmethod
    def verify_email_token(token):
        s = Serializer(app.config["SECRET_KEY"])
        try:
            user_data = s.loads(token)
            return user_data
        except:
            return None

    def __repr__(self):
        return f"User({self.id}, {self.username}, {self.email})"