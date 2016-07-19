{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}

   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
   {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}
