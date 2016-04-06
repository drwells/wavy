#include <deal.II/base/function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

constexpr int dim {2};

namespace
{
  using namespace dealii;
  class PushForward : public Function<dim>
  {
  public:
    PushForward() : Function<dim>(dim)
    {}

    double value(const Point<dim> &point,
                 const unsigned int component = 0) const override
    {
      switch (component)
        {
        case 0:
          return point[0];
        case 1:
          return 1.0 + 0.2*std::sin(numbers::PI*get_time())*std::sin(2.0*numbers::PI*point[0]);
        default:
          Assert(false, ExcNotImplemented());
        }
      return std::numeric_limits<double>::quiet_NaN();
    }
  };


  class PullBack : public Function<dim>
  {
  public:
    PullBack() : Function<dim>(dim)
    {}

    double value(const Point<dim> &point,
                 const unsigned int component = 0) const override
    {
      switch (component)
        {
        case 0:
          return point[0];
        case 1:
          {
            const auto time = get_time();
            if (time > 0.0)
              {
                return 1.0;
              }
            else
              {
                Assert(time == 0.0, ExcNotImplemented());
                return 0.0;
              }
          }
        default:
          Assert(false, ExcNotImplemented());
        }
      return std::numeric_limits<double>::quiet_NaN();
    }
  };
}

/**
 * Enforce the desired invariant that all cells are subdivided by bisecting
 * faces. Only performs this action in the y direction.
 */
void enforce_bisection_invariant(Triangulation<2>::cell_iterator &cell,
                                 const bool recur)
{
  if (!cell->active())
    {
      const auto left_bottom_child = cell->child(0);
      Assert(left_bottom_child->vertex(0) == cell->vertex(0),
             ExcMessage("This should be the bottom left cell."));
      left_bottom_child->vertex(2)[1] = 0.5*(cell->vertex(0)[1] + cell->vertex(2)[1]);
      const auto right_bottom_child = cell->child(1);
      right_bottom_child->vertex(3)[1] = 0.5*(cell->vertex(1)[1] + cell->vertex(3)[1]);
      Assert(right_bottom_child->vertex(1) == cell->vertex(1),
             ExcMessage("This should be the bottom right cell."));

      // and fix the middle:
      const auto left_top_child = cell->child(2);
      Assert(left_top_child->vertex(2) == cell->vertex(2),
             ExcMessage("This should be the top left cell."));
      // bottom middle
      left_bottom_child->vertex(1)[1] = 0.5*(left_bottom_child->vertex(0)[1]
                                             + right_bottom_child->vertex(1)[1]);

      // middle middle
      left_bottom_child->vertex(3)[1] = 0.5*(left_top_child->vertex(3)[1]
                                             + left_bottom_child->vertex(1)[1]);
      // top middle, if not at roof
      if (!left_top_child->face(3)->at_boundary())
        {
          left_top_child->vertex(3)[1] = 0.5*(cell->vertex(2)[1] + cell->vertex(3)[1]);
        }
      for (unsigned int child_n = 0; child_n < GeometryInfo<dim>::max_children_per_cell;
           ++child_n)
        {
          auto child = cell->child(child_n);
          if (recur)
            {
              enforce_bisection_invariant(child, true);
            }
        }
    }
}


void mark_all_children_for_coarsening(Triangulation<dim>::cell_iterator cell)
{
  if (cell->active())
    {
      cell->set_coarsen_flag();
      return;
    }
  else
    {
      for (unsigned int child_n = 0;
           child_n < GeometryInfo<dim>::max_children_per_cell;
           ++child_n)
        {
          mark_all_children_for_coarsening(cell->child(child_n));
        }
    }
}


int main()
{
  using namespace dealii;

  // the manifold must outlive the triangulation
  PushForward forward;
  PullBack backward;
  FunctionManifold<dim> roof_manifold(forward, backward);

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  constexpr types::manifold_id roof_id {1};

  triangulation.begin_active()->face(3)->set_manifold_id(roof_id);
  triangulation.set_manifold(roof_id, roof_manifold);
  triangulation.refine_global(3);

  // refine the roof and coarsen the floor
  for (unsigned int i = 0; i < 2; ++i)
    {
      for (auto cell : triangulation.active_cell_iterators())
        {
          if (cell->at_boundary())
            {
              for (unsigned int face_n = 0;
                   face_n < GeometryInfo<dim>::faces_per_cell;
                   ++face_n)
                {
                  const auto face = cell->face(face_n);
                  if (face->manifold_id() == roof_id)
                    {
                      cell->set_refine_flag();
                    }
                  else if (face->vertex(0)[1] == 0.0)
                    {
                      auto parent = cell->parent();
                      mark_all_children_for_coarsening(parent);
                    }
                }
            }
        }
      triangulation.execute_coarsening_and_refinement();
    }

  double current_time {0.0};
  constexpr double final_time {10.0};
  constexpr double time_step {0.01};

  unsigned int time_step_n = 0;
  while (current_time < final_time)
    {
      GridOut grid_out;
      std::ofstream out_stream("tria" + Utilities::to_string(time_step_n) + ".vtu");
      grid_out.write_vtu(triangulation, out_stream);

      current_time += time_step;

      // raise the roof
      for (auto cell : triangulation.active_cell_iterators())
        {
          // move the cell vertices
          if (cell->at_boundary())
            {
              for (unsigned int face_n = 0;
                   face_n < GeometryInfo<dim>::faces_per_cell;
                   ++face_n)
                {
                  const auto face = cell->face(face_n);
                  if (face->manifold_id() == roof_id)
                    {
                      Assert(cell->level() > 0,
                             ExcMessage("Code assumes cell has a parent."));
                      Assert(face_n == 3, ExcMessage("Code assumes face_n = 3 in"
                                                     " other numberings."));
                      const auto left_height = cell->vertex(2)[1] - cell->vertex(0)[1];
                      Assert(left_height > 0, ExcMessage("height should be nonzero"));
                      const auto right_height = cell->vertex(3)[1] - cell->vertex(1)[1];
                      Assert(right_height > 0, ExcMessage("height should be nonzero"));

                      auto &top_left_vertex = face->vertex(0);
                      const auto previous_top_left_y = top_left_vertex[1];
                      Assert(top_left_vertex == cell->vertex(2),
                             ExcMessage("Code assumes conventional numbering "
                                        "everywhere."));
                      top_left_vertex[1] = forward.value(top_left_vertex, 1);
                      const auto left_change = top_left_vertex[1] - previous_top_left_y;

                      auto &top_right_vertex = face->vertex(1);
                      const auto previous_top_right_y = top_right_vertex[1];
                      Assert(top_right_vertex == cell->vertex(3),
                             ExcMessage("Code assumes conventional numbering "
                                        "everywhere."));
                      top_right_vertex[1] = forward.value(top_right_vertex, 1);
                      const auto right_change = top_right_vertex[1] - previous_top_right_y;

                      // fix the bottom vertices to preserve the invariant
                      // that all parents are divided at bisectors of lines
                      auto &bottom_left_vertex = cell->vertex(0);
                      bottom_left_vertex[1] += left_change/2.0;
                      auto &bottom_right_vertex = cell->vertex(1);
                      bottom_right_vertex[1] += right_change/2.0;
                    }
                }
            }
        }

      // preserve the invariant that all cells are refined at bisectors for
      // the layer at level two and up at the roof
      for (auto cell : triangulation.cell_iterators_on_level(1))
        {
          if (cell->at_boundary())
            {
              for (unsigned int face_n = 0;
                   face_n < GeometryInfo<dim>::faces_per_cell;
                   ++face_n)
                {
                  if (cell->face(face_n)->manifold_id() == roof_id)
                    {
                      // preserve invariant
                      enforce_bisection_invariant(cell, true);
                    }
                }
            }
        }
      forward.advance_time(time_step);
      ++time_step_n;
    }
}
