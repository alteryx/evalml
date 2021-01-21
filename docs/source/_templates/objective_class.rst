{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   .. Class attributes:

   .. autoattribute:: name
   .. autoattribute:: greater_is_better
   .. autoattribute:: perfect_score
   .. autoattribute:: positive_only
   .. autoattribute:: problem_types
   .. autoattribute:: score_needs_proba
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree: methods

   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

Class Inheritance
"""""""""""""""""

.. inheritance-diagram:: {{ objname }}
