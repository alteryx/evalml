{% if obj.display %}
{% if obj.docstring|length > 0 %}
* - {{ obj.name }}
  - {{ obj.docstring.replace("\n", "") }}
{% else %}
* - {{ obj.name }}
  - {{ obj.value|string }}
{% endif %}
{% endif %}
