from django.contrib import admin
from django.apps import apps

# Get the app config for your app 
app_config = apps.get_app_config('myapp')

# Iterate through models in the app
for model in app_config.get_models():
    # Create a custom admin class dynamically
    admin_class = type(
        f'{model.__name__}Admin',
        (admin.ModelAdmin,),
        {
            'list_display': [field.name for field in model._meta.fields],
            'list_editable': [field.name for field in model._meta.fields if field.name != 'id'],  # Exclude 'id' field
            'list_display_links': ['id'],  # Make the 'id' field clickable
        }
    )

    # Register the model with the custom admin class
    admin.site.register(model, admin_class)