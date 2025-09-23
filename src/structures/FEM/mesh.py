from structures.FEM.plate_element import CompositeElement, Element, Node


class Mesh:
    def __init__(self, nodes: list[Node], elements: list[Element]):
        self.nodes: list[Node] = nodes
        self.elements: list[CompositeElement] = elements
        self.no_dof = sum([node.dof_per_node for node in self.nodes])
        self.dof_per_node = self.nodes[0].dof_per_node

    def get_node_from_id(self, index: int) -> Node:
        return next(node for node in self.nodes if node.id == index)

    def get_id_from_node(self, node: Node) -> int:
        return self.nodes.index(node)

    def get_element_from_id(self, index: int) -> Element:
        return next(elem for elem in self.elements if elem.id == index)

    def get_id_from_element(self, element: Element) -> int:
        return self.elements.index(element)

    def __repr__(self):
        return f"Mesh(nodes={len(self.nodes)}, elements={len(self.elements)})"
