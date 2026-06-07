from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('login/',  views.login_view,  name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Pages
    path('dashboard/',           views.dashboard,        name='dashboard'),
    path('upload/',              views.upload_file,       name='upload'),
    path('files/',               views.file_list,         name='file_list'),
    path('files/<int:pk>/delete/', views.delete_file,    name='delete_file'),
    path('viewer/<int:pk>/',     views.viewer,            name='viewer'),
    path('corrections/<int:pk>/', views.corrections_view, name='corrections'),
    path('analysis/<int:pk>/',   views.analysis_view,     name='analysis'),

    # API endpoints
    path('api/band-image/<int:pk>/',      views.api_band_image,       name='api_band_image'),
    path('api/rgb-image/<int:pk>/',       views.api_rgb_image,        name='api_rgb_image'),
    path('api/spectral-profile/<int:pk>/', views.api_spectral_profile, name='api_spectral_profile'),
    path('api/save-profile/<int:pk>/',    views.api_save_profile,     name='api_save_profile'),
    path('api/profiles-list/<int:pk>/',   views.api_profiles_list,    name='api_profiles_list'),
    path('api/correct/<int:pk>/',         views.api_run_correction,   name='api_correct'),
    path('api/index/<int:pk>/',           views.api_compute_index,    name='api_index'),
    path('api/band-stats/<int:pk>/',      views.api_band_stats,       name='api_band_stats'),
    path('api/pca/<int:pk>/',             views.api_pca,              name='api_pca'),
    path('api/structure/<int:pk>/',       views.api_file_structure,   name='api_structure'),

    # Advanced Analysis API
    path('api/ppi/<int:pk>/',             views.api_run_ppi,          name='api_ppi'),
    path('api/sam/<int:pk>/',             views.api_run_sam,          name='api_sam'),
    path('api/lsu/<int:pk>/',             views.api_run_lsu,          name='api_lsu'),
    path('api/classify/<int:pk>/',        views.api_run_classification, name='api_classify'),
    path('api/cluster/<int:pk>/',         views.api_run_clustering,   name='api_cluster'),
]
