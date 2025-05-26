class message:
    def create(user, message):
        return {
            "role": user,
            "content": message
        },