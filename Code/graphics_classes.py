class Camera:
    def __init__(self,weight,height,x=0,y=0):
        self.w = weight
        self.h = height
        self.x = x
        self.y = y
    def move_left(self, step = 1):  self.x -= step
    def move_right(self, step = 1): self.x += step
    def move_up(self,step = 1):     self.y -= step
    def move_down(self, step = 1):  self.y += step


from shapely.geometry import Polygon
def check_intersection(poly1, poly2):
    # Convertir les listes de points en objets Polygon de Shapely
    shapely_poly1 = Polygon(poly1)
    shapely_poly2 = Polygon(poly2)
    
    # Vérifier si les polygones s'intersectent
    return shapely_poly1.intersects(shapely_poly2)
