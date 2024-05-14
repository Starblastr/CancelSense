from django import template

register = template.Library()

@register.filter(name='multiply')
def multiply(value, arg):
    """Multiplies the value by arg. Usage: {{ value|multiply:arg }}"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''  # You might choose to handle the error differently.
