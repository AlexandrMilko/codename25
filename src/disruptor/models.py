from disruptor import db, login_manager
from flask_login import UserMixin

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

    def __repr__(self):
        return f"User({self.id}, {self.username}, {self.email})"