from disruptor import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"User({self.id}, {self.username}, {self.email})"